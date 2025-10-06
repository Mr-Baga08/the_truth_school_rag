# Enhanced RAG-Anything Backend API

Production-ready FastAPI backend for the RAG-Anything system with multi-domain support and advanced AI features.

## Features

### 🎯 Multi-Domain Support
- **Medical & Healthcare**: Medical documents, research papers, clinical guidelines
- **Legal & Compliance**: Legal documents, contracts, regulations, case law
- **Financial & Analytics**: Financial reports, analysis, market research
- **Technical Documentation**: Technical docs, APIs, code, architecture
- **Academic Research**: Research papers, academic publications, studies

### 🚀 Advanced AI Capabilities
- **Query Improvement**: Automatic query enhancement with abbreviation expansion
- **Dual-LLM Verification**: Two-stage answer verification for quality assurance
- **Conversation Memory**: Context-aware responses with conversation history
- **Multimodal Processing**: Support for images, tables, and equations
- **Domain-Specific Prompts**: Optimized prompts for each domain

### 🔧 Technical Features
- **Gemini API Integration**: Free-tier Gemini 1.5 Flash model
- **Async Processing**: Background document processing
- **RESTful API**: Clean, well-documented endpoints
- **CORS Support**: Cross-origin resource sharing enabled
- **Error Handling**: Comprehensive error handling and logging

## Installation

### Prerequisites
- Python 3.9+
- Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Setup

1. **Clone the repository**
```bash
cd /mnt/data/Agentic_RAG/backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```env
GEMINI_API_KEY=your-api-key-here
```

4. **Run the server**
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-04T10:00:00",
  "version": "1.0.0",
  "features": {
    "query_improvement": true,
    "dual_llm_verification": true,
    "conversation_memory": true,
    "multi_domain": true,
    "multimodal_processing": true,
    "gemini_integration": true
  },
  "domains": ["medical", "legal", "financial", "technical", "academic"]
}
```

### List Domains
```bash
GET /domains
```

### Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

file: <document file>
domain: medical
```

Response:
```json
{
  "success": true,
  "message": "Document uploaded and queued for processing",
  "file_name": "research_paper.pdf",
  "domain": "medical",
  "processing_id": "uuid-here"
}
```

### Query Documents
```bash
POST /query
Content-Type: application/json

{
  "query": "What are the treatment options for hypertension?",
  "domain": "medical",
  "mode": "mix",
  "conversation_id": "conv_123",
  "return_metadata": true
}
```

Response:
```json
{
  "answer": "Hypertension treatment includes lifestyle modifications...",
  "sources": ["medical_guidelines.pdf"],
  "confidence_score": 0.92,
  "query_improved": true,
  "verification_performed": true,
  "conversation_id": "conv_123",
  "metadata": {
    "original_query": "What is HTN treatment?",
    "improved_query": "What are the treatment options for hypertension?",
    "verification_score": 8.5,
    "modification_attempts": 1
  }
}
```

### Get Conversation History
```bash
GET /conversation/{conversation_id}
```

### Clear Conversation
```bash
DELETE /conversation/{conversation_id}
```

### Clear Domain Data
```bash
DELETE /clear/{domain}
```

## Usage Examples

### Using cURL

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@medical_paper.pdf" \
  -F "domain=medical"
```

**Query documents:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the side effects of ACE inhibitors?",
    "domain": "medical",
    "mode": "mix",
    "return_metadata": true
  }'
```

### Using Python

```python
import requests

# Upload document
with open("medical_paper.pdf", "rb") as f:
    files = {"file": f}
    data = {"domain": "medical"}
    response = requests.post("http://localhost:8000/upload", files=files, data=data)
    print(response.json())

# Query documents
query_data = {
    "query": "What are the treatment options for hypertension?",
    "domain": "medical",
    "mode": "mix",
    "return_metadata": True
}
response = requests.post("http://localhost:8000/query", json=query_data)
print(response.json())
```

### Using JavaScript/TypeScript

```typescript
// Upload document
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('domain', 'medical');

const uploadResponse = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});

// Query documents
const queryResponse = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'What are the treatment options for hypertension?',
    domain: 'medical',
    mode: 'mix',
    return_metadata: true
  })
});

const result = await queryResponse.json();
console.log(result);
```

## Configuration

### Domain-Specific Settings

Each domain has customized settings in `DOMAIN_CONFIGS`:

```python
{
    "medical": {
        "enable_query_improvement": True,
        "query_improvement_method": "hybrid",
        "expand_abbreviations": True,
        "verification_threshold": 7.5,
        # ... more settings
    }
}
```

### Gemini Model Configuration

Currently using `gemini-1.5-flash` (free tier). To use a different model:

```python
GEMINI_MODEL = "gemini-1.5-pro"  # More capable, paid tier
```

## Architecture

```
backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # This file

storage/                 # Created at runtime
├── medical/            # Medical domain storage
├── legal/              # Legal domain storage
├── financial/          # Financial domain storage
├── technical/          # Technical domain storage
└── academic/           # Academic domain storage

uploads/                # Uploaded files
├── medical/
├── legal/
└── ...
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `500`: Internal Server Error

All errors return JSON:
```json
{
  "detail": "Error message here"
}
```

## Logging

Logs are output to console with the format:
```
2025-01-04 10:00:00 - main - INFO - Message here
```

## Production Deployment

For production deployment:

1. **Set proper CORS origins** in `main.py`:
```python
allow_origins=["https://your-frontend-domain.com"]
```

2. **Use a production ASGI server**:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

3. **Set up environment variables** securely (don't commit `.env` files)

4. **Enable HTTPS** using a reverse proxy (nginx, Caddy, etc.)

5. **Set up proper logging** (file-based, log rotation)

6. **Monitor** with tools like Prometheus, Grafana

## Troubleshooting

### "GEMINI_API_KEY not set"
Set your API key as an environment variable or in a `.env` file.

### "Failed to initialize RAG system"
Check that the storage directories are writable and all dependencies are installed.

### "File type not supported"
Verify the file extension is in the allowed list for the target domain.

## License

[Your License Here]

## Support

For issues and questions, please open an issue on GitHub.
