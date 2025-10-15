# Hugging Face Docker Space Dockerfile
# This combines backend and frontend into a single container for Hugging Face Spaces
# Exposes port 7860 as required by Hugging Face

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    nginx \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy rag_anything_smaranika first to install as package
COPY rag_anything_smaranika/ /app/rag_anything_smaranika/

# Install rag_anything_smaranika package
WORKDIR /app/rag_anything_smaranika
RUN pip install --no-cache-dir -e .

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    lightrag-hku \
    cachetools \
    aiofiles

# Copy backend code
COPY backend/ /app/backend/

# Copy frontend code
COPY frontend/ /app/frontend/

# Build frontend
WORKDIR /app/frontend
RUN npm install
RUN REACT_APP_BACKEND_URL=/api npm run build

# Configure nginx to serve frontend and proxy backend
RUN echo 'server { \
    listen 7860; \
    server_name _; \
    \
    # Serve frontend \
    location / { \
        root /app/frontend/build; \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # Proxy backend API \
    location /api/ { \
        proxy_pass http://127.0.0.1:8000/; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        proxy_buffering off; \
        proxy_cache_bypass $http_upgrade; \
    } \
    \
    # Health check endpoint \
    location /health { \
        proxy_pass http://127.0.0.1:8000/health; \
        proxy_http_version 1.1; \
        proxy_set_header Host $host; \
    } \
}' > /etc/nginx/sites-available/default

# Create necessary directories
RUN mkdir -p /app/storage /app/uploads /app/backend/output

# Set working directory back to /app
WORKDIR /app

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Agentic RAG System for Hugging Face Space..."\n\
\n\
# Check for required environment variables\n\
if [ -z "$GEMINI_API_KEY" ]; then\n\
    echo "ERROR: GEMINI_API_KEY environment variable is not set!"\n\
    echo "Please set it in your Hugging Face Space settings."\n\
    exit 1\n\
fi\n\
\n\
# Start backend in background\n\
echo "Starting FastAPI backend on port 8000..."\n\
cd /app\n\
export PYTHONPATH=/app:$PYTHONPATH\n\
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info &\n\
BACKEND_PID=$!\n\
\n\
# Wait for backend to be ready\n\
echo "Waiting for backend to be ready..."\n\
for i in {1..30}; do\n\
    if curl -s http://127.0.0.1:8000/health > /dev/null; then\n\
        echo "Backend is ready!"\n\
        break\n\
    fi\n\
    echo "Waiting for backend... ($i/30)"\n\
    sleep 2\n\
done\n\
\n\
# Start nginx\n\
echo "Starting nginx on port 7860..."\n\
nginx -g "daemon off;" &\n\
NGINX_PID=$!\n\
\n\
echo "==========================================="\n\
echo "Agentic RAG System is running!"\n\
echo "Backend: http://localhost:8000"\n\
echo "Frontend: http://localhost:7860"\n\
echo "API Docs: http://localhost:8000/docs"\n\
echo "==========================================="\n\
\n\
# Wait for both processes\n\
wait $BACKEND_PID $NGINX_PID\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port 7860 (Hugging Face Space requirement)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV BACKEND_PORT=8000
ENV FRONTEND_PORT=7860

# Start the application
CMD ["/app/start.sh"]
