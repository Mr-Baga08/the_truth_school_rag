---
title: Agentic RAG System
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---
  
# 🤖 Agentic RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system with multi-domain support, advanced AI features, and intelligent web search integration. Built with FastAPI, React, and Google Gemini API.

## ✨ Features

### 🎯 Multi-Domain Support
- **Medical & Healthcare**: Medical documents, research papers, clinical guidelines
- **Legal & Compliance**: Legal documents, contracts, regulations, case law
- **Financial & Analytics**: Financial reports, analysis, market research
- **Technical Documentation**: Technical docs, APIs, code, architecture
- **Academic Research**: Research papers, academic publications, studies

### 🚀 Advanced AI Capabilities
- **Query Improvement**: Automatic query enhancement with abbreviation expansion
- **Dual-LLM Verification**: Two-stage answer verification using Gemini Pro
- **Web Search Integration**: Augment answers with real-time web search via Tavily
- **Conversation Memory**: Context-aware responses with conversation history
- **Multimodal Processing**: Support for images, tables, and equations (MinerU parser)
- **Smart Reranking**: Gemini-powered relevance reranking for better results
- **Streaming Responses**: Real-time token streaming for responsive UX

### 🔧 Technical Features
- **Gemini API Integration**: Free-tier Gemini Flash & Pro models
- **Async Processing**: Background document processing with status tracking
- **RESTful API**: Clean, well-documented FastAPI endpoints
- **Modern React Frontend**: Beautiful, responsive UI with Tailwind CSS
- **Docker Support**: One-command deployment with docker-compose
- **Performance Optimized**: Query caching, fast mode (2-3x speedup), batch processing

## 🚀 Quick Start (Docker)

### Prerequisites
- Docker and Docker Compose
- Google Gemini API Key ([Get one free](https://makersuite.google.com/app/apikey))
- (Optional) Tavily API Key for web search ([Get one free](https://tavily.com))

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Agentic_RAG
```

### 2. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_TEXT_MODEL=models/gemini-flash-latest
GEMINI_VERIFIER_MODEL=models/gemini-pro-latest
GEMINI_VISION_MODEL=models/gemini-flash-latest
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
TAVILY_API_KEY=your_tavily_api_key_here  # Optional, for web search
```

### 3. Start the Application
```bash
docker-compose up -d
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📖 Usage

### Upload Documents
1. Navigate to the frontend at http://localhost:3000
2. Select a domain (Medical, Legal, Financial, Technical, or Academic)
3. Upload PDF, DOCX, TXT, or other supported documents
4. Wait for processing to complete (tracked with real-time status)

### Query Documents
1. Enter your question in the query interface
2. Select query mode:
   - **Mix**: Balanced combination of local and global search (recommended)
   - **Local**: Focused chunk-based search
   - **Global**: Knowledge graph entity search
   - **Hybrid**: Advanced combination
   - **Web**: RAG + real-time web search
3. Toggle advanced features:
   - **Query Improvement**: Enhance your query automatically
   - **Verification**: Dual-LLM quality check
   - **Web Search**: Augment with real-time web results
   - **Fast Mode**: 2-3x faster queries (slightly lower quality)
4. Get streaming responses with sources and confidence scores

### API Usage
```python
import requests

# Upload document
files = {"file": open("document.pdf", "rb")}
data = {"domain": "medical"}
response = requests.post("http://localhost:8000/upload", files=files, data=data)
print(response.json())

# Query documents
query_data = {
    "query": "What are the treatment options for hypertension?",
    "domain": "medical",
    "mode": "mix",
    "enable_web_search": False,
    "fast_mode": False,
    "return_metadata": True
}
response = requests.post("http://localhost:8000/query", json=query_data)
print(response.json())
```

## 🏗️ Architecture

```
Agentic_RAG/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API server
│   ├── reranker.py         # Gemini-powered reranking
│   ├── web_search.py       # Tavily web search integration
│   ├── url_fetcher.py      # URL content fetching
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container
├── frontend/               # React frontend
│   ├── src/               # React components
│   ├── public/            # Static assets
│   ├── package.json       # Node dependencies
│   ├── Dockerfile         # Frontend container
│   └── nginx.conf         # Nginx configuration
├── storage/               # RAG storage (created at runtime)
│   ├── medical/          # Medical domain storage
│   ├── legal/            # Legal domain storage
│   └── ...               # Other domains
├── uploads/              # Uploaded documents
├── docker-compose.yml    # Docker orchestration
├── Dockerfile            # Hugging Face Space Dockerfile
└── README.md            # This file
```

## 🔑 Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes | - |
| `GEMINI_TEXT_MODEL` | Text generation model | No | `models/gemini-flash-latest` |
| `GEMINI_VERIFIER_MODEL` | Verification model | No | `models/gemini-pro-latest` |
| `GEMINI_VISION_MODEL` | Vision processing model | No | `models/gemini-flash-latest` |
| `GEMINI_EMBEDDING_MODEL` | Embedding model | No | `models/text-embedding-004` |
| `TAVILY_API_KEY` | Tavily web search API key | No | - |

## 📊 API Endpoints

### Health Check
```bash
GET /health
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

### Query Documents (Streaming)
```bash
POST /query/stream
Content-Type: application/json

{
  "query": "What are the treatment options?",
  "domain": "medical",
  "mode": "mix",
  "enable_web_search": false,
  "fast_mode": false
}
```

### Query Documents (Standard)
```bash
POST /query
Content-Type: application/json

{
  "query": "What are the treatment options?",
  "domain": "medical",
  "mode": "mix"
}
```

### Check Processing Status
```bash
GET /status/{processing_id}
```

### List Documents
```bash
GET /documents?domain=medical
```

### Delete Document
```bash
DELETE /documents/{doc_id}
```

## 🎯 Performance

- **Fast Mode**: 2-3x faster queries with optimized parameters
- **Query Caching**: 5-minute TTL cache for repeated queries
- **Batch Processing**: Parallel document processing (up to 10 documents)
- **Streaming**: Real-time token streaming for responsive UX
- **Reranking**: Gemini-powered relevance scoring

## 🛠️ Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Development
```bash
cd frontend
npm install
npm start
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LightRAG](https://github.com/HKUDS/LightRAG) - RAG framework
- [Google Gemini](https://ai.google.dev/) - LLM and embeddings
- [Tavily](https://tavily.com/) - Web search API
- [MinerU](https://github.com/opendatalab/MinerU) - Document parsing
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [React](https://react.dev/) - Frontend framework

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

Built with ❤️ using FastAPI, React, and Google Gemini
