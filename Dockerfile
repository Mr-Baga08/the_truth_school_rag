
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

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Install raganything dependencies (if not in requirements.txt)
RUN pip install --no-cache-dir \
    lightrag-hku \
    cachetools \
    aiofiles

# Copy backend code
COPY backend/ /app/backend/
COPY rag_anything_smaranika/ /app/rag_anything_smaranika/

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


# Expose port 7860 (Hugging Face Space requirement)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BACKEND_PORT=8000
ENV FRONTEND_PORT=7860

# Start the application
CMD ["/app/start.sh"]
