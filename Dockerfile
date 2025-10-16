
FROM python:3.12-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    nginx \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*


COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --use-pep517 -r /app/backend/requirements.txt

# Copy local modified LightRAG package in vendor directory
COPY vendor/ /app/vendor/

COPY backend/ /app/backend/
COPY rag_anything_smaranika/ /app/rag_anything_smaranika/


COPY frontend/ /app/frontend/


WORKDIR /app/frontend
RUN npm install
RUN REACT_APP_API_URL=/api npm run build



RUN mkdir -p /var/lib/nginx/body /var/lib/nginx/fastcgi \
    /var/lib/nginx/proxy /var/lib/nginx/scgi /var/lib/nginx/uwsgi \
    /var/log/nginx /var/cache/nginx && \
    chmod -R 777 /var/lib/nginx /var/log/nginx /var/cache/nginx && \
    touch /var/run/nginx.pid && chmod 666 /var/run/nginx.pid


RUN echo 'pid /tmp/nginx.pid;\n\
error_log /var/log/nginx/error.log;\n\
events {\n\
    worker_connections 1024;\n\
}\n\
http {\n\
    include /etc/nginx/mime.types;\n\
    default_type application/octet-stream;\n\
    access_log /var/log/nginx/access.log;\n\
    client_body_temp_path /tmp/client_body;\n\
    proxy_temp_path /tmp/proxy;\n\
    fastcgi_temp_path /tmp/fastcgi;\n\
    uwsgi_temp_path /tmp/uwsgi;\n\
    scgi_temp_path /tmp/scgi;\n\
    \n\
    server {\n\
        listen 7860;\n\
        server_name _;\n\
        \n\
        location / {\n\
            root /app/frontend/build;\n\
            try_files $uri $uri/ /index.html;\n\
        }\n\
        \n\
        location /api/ {\n\
            proxy_pass http://127.0.0.1:8000/;\n\
            proxy_http_version 1.1;\n\
            proxy_set_header Upgrade $http_upgrade;\n\
            proxy_set_header Connection "upgrade";\n\
            proxy_set_header Host $host;\n\
            proxy_set_header X-Real-IP $remote_addr;\n\
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
            proxy_set_header X-Forwarded-Proto $scheme;\n\
            proxy_buffering off;\n\
            proxy_cache_bypass $http_upgrade;\n\
        }\n\
        \n\
        location /health {\n\
            proxy_pass http://127.0.0.1:8000/health;\n\
            proxy_http_version 1.1;\n\
            proxy_set_header Host $host;\n\
        }\n\
    }\n\
}' > /etc/nginx/nginx.conf


RUN mkdir -p /app/storage /app/uploads /app/backend/output /app/output /app/.cache/huggingface && \
    chmod -R 777 /app/storage /app/uploads /app/backend/output /app/output /app/.cache

WORKDIR /app

WORKDIR /app/rag_anything_smaranika
RUN pip install --no-cache-dir -e .

WORKDIR /app

RUN mkdir -p /app/storage/medical /app/storage/legal /app/storage/financial \
    /app/storage/technical /app/storage/academic && \
    chmod -R 777 /app/storage

# Create output directory in the working directory for the parser
RUN mkdir -p /app/output && chmod -R 777 /app/output


RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "===== Application Startup at $(date +"%Y-%m-%d %H:%M:%S") ====="\n\
echo ""\n\
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
export PYTHONPATH=/app:/app/vendor:$PYTHONPATH\n\
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info &\n\
BACKEND_PID=$!\n\
\n\
# Wait for backend to be ready\n\
echo "Waiting for backend to be ready..."\n\
for i in {1..30}; do\n\
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then\n\
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
echo ""\n\
echo "==========================================="\n\
echo "Agentic RAG System is running!"\n\
echo "Backend: http://localhost:8000"\n\
echo "Frontend: http://localhost:7860"\n\
echo "API Docs: http://localhost:8000/docs"\n\
echo "==========================================="\n\
echo ""\n\
\n\
# Wait for both processes\n\
wait $BACKEND_PID $NGINX_PID\n\
' > /app/start.sh && chmod +x /app/start.sh


EXPOSE 7860


HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/vendor:$PYTHONPATH
ENV BACKEND_PORT=8000
ENV FRONTEND_PORT=7860
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV PYTHONOPTIMIZE=0

# Start the application
CMD ["/app/start.sh"]
