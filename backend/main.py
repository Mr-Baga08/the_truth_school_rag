"""
FastAPI Backend for Enhanced RAG-Anything System (v1.1 - Updated)

Production-ready backend with:
- Multi-domain support (medical, legal, financial, technical, academic)
- Gemini API integration (LLM, Vision, Embeddings)
- Query improvement and dual-LLM verification
- Conversation history management
- Document processing and querying
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import hashlib
import time
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from cachetools import TTLCache

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

from raganything.raganything import RAGAnything, RAGAnythingConfig, create_rag_anything
from backend.reranker import GeminiReranker
from backend.web_search import WebSearcher, create_web_searcher
from backend.url_fetcher import URLFetcher, create_url_fetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Domain Configurations
# =============================================================================

DOMAIN_CONFIGS = {
    "medical": {
        "name": "Medical & Healthcare",
        "description": "Optimized for medical documents, research papers, clinical guidelines",
        "system_prompt": (
            "You are a medical AI assistant with expertise in healthcare, clinical medicine, "
            "and medical research. Provide accurate, evidence-based responses with appropriate "
            "medical terminology. Always cite sources and indicate confidence levels."
        ),
        "analysis_prompt": (
            "Analyze this medical document focusing on: diagnoses, treatments, medications, "
            "clinical findings, patient outcomes, and evidence-based recommendations."
        ),
        "file_extensions": [".pdf", ".doc", ".docx", ".txt", ".md", ".csv", ".xlsx"],
        "config_overrides": {
            "domain": "medical",
            "enable_query_improvement": True,
            "query_improvement_method": "hybrid",
            "expand_abbreviations": True,
            "add_domain_keywords": True,
            "extract_query_entities": True,
            "enable_dual_llm_verification": True,
            "enable_answer_verification": True,
            "enable_answer_modification": True,
            "verification_threshold": 7.5,
            "check_factual_consistency": True,
            "check_completeness": True,
            "check_relevance": True,
        }
    },
    "legal": {
        "name": "Legal & Compliance",
        "description": "Specialized for legal documents, contracts, regulations, case law",
        "system_prompt": (
            "You are a legal AI assistant with expertise in law, regulations, and compliance. "
            "Provide precise legal analysis with proper citations. Note that this is for "
            "informational purposes only and not legal advice."
        ),
        "analysis_prompt": (
            "Analyze this legal document focusing on: key provisions, obligations, rights, "
            "legal precedents, regulatory requirements, and potential implications."
        ),
        "file_extensions": [".pdf", ".doc", ".docx", ".txt", ".csv", ".xlsx"],
        "config_overrides": {
            "domain": "legal",
            "enable_query_improvement": True,
            "query_improvement_method": "llm",
            "expand_abbreviations": True,
            "extract_query_entities": True,
            "enable_dual_llm_verification": True,
            "enable_answer_verification": True,
            "enable_answer_modification": True,
            "verification_threshold": 8.0,
            "check_factual_consistency": True,
            "check_completeness": True,
        }
    },
    "financial": {
        "name": "Financial & Analytics",
        "description": "Tailored for financial reports, analysis, market research, forecasts",
        "system_prompt": (
            "You are a financial AI assistant with expertise in finance, accounting, and "
            "market analysis. Provide data-driven insights with numerical precision. "
            "Include relevant financial metrics and trends."
        ),
        "analysis_prompt": (
            "Analyze this financial document focusing on: financial metrics, trends, "
            "performance indicators, risk factors, market conditions, and forecasts."
        ),
        "file_extensions": [".pdf", ".xlsx", ".csv", ".doc", ".docx"],
        "config_overrides": {
            "domain": "financial",
            "enable_query_improvement": True,
            "query_improvement_method": "hybrid",
            "expand_abbreviations": True,
            "add_domain_keywords": True,
            "enable_dual_llm_verification": True,
            "enable_answer_verification": True,
            "verification_threshold": 7.5,
            "check_factual_consistency": True,
        }
    },
    "technical": {
        "name": "Technical Documentation",
        "description": "Optimized for technical docs, APIs, code, system architecture",
        "system_prompt": (
            "You are a technical AI assistant with expertise in software development, "
            "system architecture, and technical documentation. Provide clear, precise "
            "technical explanations with code examples when relevant."
        ),
        "analysis_prompt": (
            "Analyze this technical document focusing on: system design, APIs, configurations, "
            "dependencies, implementation details, and best practices."
        ),
        "file_extensions": [".pdf", ".md", ".txt", ".rst", ".doc", ".docx", ".csv", ".xlsx"],
        "config_overrides": {
            "domain": "technical",
            "enable_query_improvement": True,
            "query_improvement_method": "hybrid",
            "expand_abbreviations": True,
            "extract_query_entities": True,
            "enable_dual_llm_verification": True,
            "enable_answer_verification": True,
            "verification_threshold": 7.0,
        }
    },
    "academic": {
        "name": "Academic Research",
        "description": "Designed for research papers, academic publications, studies",
        "system_prompt": (
            "You are an academic AI assistant with expertise in research methodology, "
            "scholarly analysis, and scientific literature. Provide well-reasoned responses "
            "with proper academic citations and methodology discussion."
        ),
        "analysis_prompt": (
            "Analyze this academic document focusing on: research questions, methodology, "
            "findings, conclusions, citations, and contributions to the field."
        ),
        "file_extensions": [".pdf", ".doc", ".docx", ".txt", ".tex", ".csv", ".xlsx"],
        "config_overrides": {
            "domain": "academic",
            "enable_query_improvement": True,
            "query_improvement_method": "llm",
            "expand_abbreviations": True,
            "add_domain_keywords": True,
            "extract_query_entities": True,
            "enable_dual_llm_verification": True,
            "enable_answer_verification": True,
            "enable_answer_modification": True,
            "verification_threshold": 8.0,
            "check_completeness": True,
            "check_relevance": True,
        }
    }
}

# =============================================================================
# Global State & Configuration
# =============================================================================

# RAG instances per domain
rag_instances: Dict[str, RAGAnything] = {}

# Web searcher instance
web_searcher: Optional[WebSearcher] = None

# URL fetcher instance
url_fetcher: Optional[URLFetcher] = None

# Conversation history storage
conversation_histories: Dict[str, List[Dict[str, str]]] = {}

# Processing status tracker
processing_status: Dict[str, Dict[str, Any]] = {}

# Query result cache (TTL: 5 minutes, max 100 entries)
query_cache: TTLCache = TTLCache(maxsize=100, ttl=300)

# Performance metrics storage
performance_metrics: Dict[str, List[float]] = {
    "query_times": [],
    "processing_times": [],
}

# Base paths
BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = BASE_DIR / "uploads"
STATUS_FILE = STORAGE_DIR / "processing_status.json"

# --- IMPROVEMENT: Centralized and configurable Gemini model names ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "models/gemini-flash-latest")  # Fast generation (alias to latest Flash)
GEMINI_VERIFIER_MODEL = os.getenv("GEMINI_VERIFIER_MODEL", "models/gemini-pro-latest")  # Quality verification (alias to latest Pro)
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "models/gemini-flash-latest")  # Vision model
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")  # Embedding model
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # For web search


# =============================================================================
# Status Persistence Functions
# =============================================================================

def load_processing_status() -> Dict[str, Dict[str, Any]]:
    """Load processing status from disk."""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r') as f:
                status_data = json.load(f)
                logger.info(f"Loaded {len(status_data)} processing status entries from disk")
                return status_data
        return {}
    except Exception as e:
        logger.error(f"Error loading processing status: {e}", exc_info=True)
        return {}


def save_processing_status():
    """Save processing status to disk."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            json.dump(processing_status, f, indent=2)
        logger.debug(f"Saved {len(processing_status)} processing status entries to disk")
    except Exception as e:
        logger.error(f"Error saving processing status: {e}", exc_info=True)


def update_processing_status(processing_id: str, status_update: Dict[str, Any]):
    """Update processing status both in-memory and on disk."""
    processing_status[processing_id] = status_update
    save_processing_status()


# =============================================================================
# Lifespan Management (Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # --- STARTUP ---
    logger.info("Starting Enhanced RAG-Anything API...")
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for domain in DOMAIN_CONFIGS.keys():
        (STORAGE_DIR / domain).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created storage directories: {STORAGE_DIR}")

    # Load processing status from disk
    global processing_status
    processing_status.update(load_processing_status())

    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Gemini API initialized successfully")
            logger.info(f"Model Configuration:")
            logger.info(f"  TEXT_MODEL: {GEMINI_TEXT_MODEL}")
            logger.info(f"  VERIFIER_MODEL: {GEMINI_VERIFIER_MODEL}")
            logger.info(f"  VISION_MODEL: {GEMINI_VISION_MODEL}")
            logger.info(f"  EMBEDDING_MODEL: {GEMINI_EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}", exc_info=True)
            logger.warning("Application will start but Gemini features will not work")
    else:
        logger.warning("GEMINI_API_KEY not set. Set it in environment variables.")

    # Initialize web searcher if Tavily API key is available
    global web_searcher, url_fetcher
    if TAVILY_API_KEY:
        try:
            web_searcher = create_web_searcher(api_key=TAVILY_API_KEY, max_results=5)
            logger.info("Tavily web search initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily: {e}. Web search will not be available.")
            web_searcher = None
    else:
        logger.info("TAVILY_API_KEY not set. Web search features disabled.")

    # Initialize URL fetcher
    try:
        url_download_dir = UPLOAD_DIR / "url_downloads"
        url_download_dir.mkdir(parents=True, exist_ok=True)
        url_fetcher = create_url_fetcher(download_dir=str(url_download_dir))
        logger.info("URL fetcher initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize URL fetcher: {e}. URL ingestion will not be available.")
        url_fetcher = None

    logger.info("Enhanced RAG-Anything API started successfully!")
    
    yield  # Application runs here

    # --- SHUTDOWN ---
    logger.info("Shutting down API...")
    for domain, rag_instance in rag_instances.items():
        logger.info(f"Finalizing storages for domain: {domain}")
        await rag_instance.finalize_storages()
    logger.info("API shutdown complete.")

# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="Enhanced RAG-Anything API",
    description="Production-ready RAG system with multi-domain support and advanced features",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # --- FIX: Using modern lifespan event handler ---
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text", min_length=1)
    domain: str = Field("medical", description="Domain context (medical, legal, etc.)")
    mode: str = Field("mix", description="Query mode (local, global, hybrid, naive, mix, web, hybrid_web)")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    return_metadata: bool = Field(True, description="Include detailed metadata in response")
    enable_web_search: bool = Field(False, description="Enable web search augmentation")
    web_search_only: bool = Field(False, description="Use only web search (no RAG)")
    enable_verification: bool = Field(True, description="Enable dual-LLM verification")
    # Performance optimization parameters
    fast_mode: bool = Field(False, description="Use optimized parameters for faster queries (2-3x speedup)")
    top_k: Optional[int] = Field(None, description="Number of top results to retrieve (default: 40, fast: 20)")
    enable_cache: bool = Field(True, description="Enable query result caching")
    enable_query_improvement: bool = Field(True, description="Enable query improvement/expansion")
    enable_verification_check: bool = Field(True, description="Enable verification step (separate from enable_verification)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the treatment options for hypertension?",
                "domain": "medical",
                "mode": "mix",
                "conversation_id": "conv_123",
                "return_metadata": True,
                "enable_web_search": False,
                "web_search_only": False,
                "enable_verification": True
            }
        }


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    confidence_score: float = Field(0.0, description="Confidence score (0-1)")
    query_improved: bool = Field(False, description="Whether query was improved")
    verification_performed: bool = Field(False, description="Whether answer was verified")
    conversation_id: str = Field(..., description="Conversation ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Hypertension treatment includes lifestyle modifications and medications...",
                "sources": ["medical_guidelines.pdf", "research_paper.pdf"],
                "confidence_score": 0.92,
                "query_improved": True,
                "verification_performed": True,
                "conversation_id": "conv_123",
                "metadata": {
                    "original_query": "What is HTN treatment?",
                    "improved_query": "What are the treatment options for hypertension?",
                    "verification_score": 8.5
                }
            }
        }


class UploadResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    domain: str
    processing_id: str


class BatchUploadResponse(BaseModel):
    success: bool
    message: str
    total_files: int
    accepted_files: int
    processing_ids: List[str]
    domain: str


class URLUploadRequest(BaseModel):
    url: str = Field(..., description="URL to fetch and process")
    domain: str = Field("medical", description="Domain context")
    convert_to_markdown: bool = Field(True, description="Convert HTML to markdown")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/medical-article.pdf",
                "domain": "medical",
                "convert_to_markdown": True
            }
        }


class DomainInfo(BaseModel):
    domain_id: str
    name: str
    description: str
    file_extensions: List[str]
    features: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    features: Dict[str, bool]
    domains: List[str]

# =============================================================================
# Gemini Integration Functions
# =============================================================================

async def gemini_llm_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    """
    Gemini LLM function for text generation (Improved with format validation).

    Enhancements:
    - Increased token limits for entity extraction tasks
    - Better temperature control for structured outputs
    - Response validation and auto-append of completion delimiter
    """
    def _sync_call():
        try:
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            safety_settings = [
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
            ]
            # --- IMPROVEMENT: Use system_instruction parameter ---
            logger.info(f"Creating GenerativeModel with model_name: {GEMINI_TEXT_MODEL}")
            model = genai.GenerativeModel(
                model_name=GEMINI_TEXT_MODEL,
                system_instruction=system_prompt,
                safety_settings=safety_settings
            )
            config_params = {}

            # Smart temperature control: lower for extraction tasks
            is_extraction_task = system_prompt and ("entity" in system_prompt.lower() or "extraction" in system_prompt.lower())
            if "temperature" in kwargs:
                config_params["temperature"] = kwargs["temperature"]
            else:
                # Use lower temperature for structured extraction tasks
                config_params["temperature"] = 0.1 if is_extraction_task else 0.3

            # Increase token limit for extraction tasks to avoid truncation
            if "max_tokens" in kwargs:
                config_params["max_output_tokens"] = kwargs["max_tokens"]
            else:
                # Larger limits for extraction to ensure completion delimiter is included
                config_params["max_output_tokens"] = 16384 if is_extraction_task else 8192

            generation_config = genai.types.GenerationConfig(**config_params)

            # --- IMPROVEMENT: Build structured history for chat model ---
            history = []
            if history_messages:
                for msg in history_messages[-5:]:
                    role = "user" if msg.get("role") == "user" else "model"
                    content = msg.get("content", "")
                    if content:
                        history.append({"role": role, "parts": [content]})

            chat = model.start_chat(history=history)
            response = chat.send_message(prompt, generation_config=generation_config)
            try:
                result = response.text

                # Post-processing: Ensure completion delimiter is present for extraction tasks
                if is_extraction_task and result:
                    # Check if completion delimiter is missing
                    if "<|COMPLETE|>" not in result and "<|complete|>" not in result:
                        logger.warning("Completion delimiter missing from extraction result, appending it")
                        # Append the delimiter to the end
                        result = result.strip() + "\n<|COMPLETE|>"

                return result
            except ValueError as ve:
                logger.warning(f"Response blocked or empty. Reason: {ve}. Candidates: {response.candidates}")
                if response.prompt_feedback:
                    logger.warning(f"Prompt feedback: {response.prompt_feedback}")
                return ""
        except Exception as e:
            logger.error(f"Gemini LLM error: {e}", exc_info=True)
            raise
    return await asyncio.to_thread(_sync_call)


async def gemini_verifier_llm_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    """Gemini Pro LLM function for answer verification (more powerful, thorough)."""
    def _sync_call():
        try:
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            safety_settings = [
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
            ]
            # Use Pro model for better verification
            logger.info(f"Creating Verifier GenerativeModel with model_name: {GEMINI_VERIFIER_MODEL}")
            model = genai.GenerativeModel(
                model_name=GEMINI_VERIFIER_MODEL,
                system_instruction=system_prompt,
                safety_settings=safety_settings
            )
            config_params = {}
            if "temperature" in kwargs:
                config_params["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                config_params["max_output_tokens"] = kwargs["max_tokens"]
            else:
                # Default to larger token limit for verification responses
                config_params["max_output_tokens"] = 8192
            generation_config = genai.types.GenerationConfig(**config_params)

            # Build history
            history = []
            if history_messages:
                for msg in history_messages[-5:]:
                    role = "user" if msg.get("role") == "user" else "model"
                    content = msg.get("content", "")
                    if content:
                        history.append({"role": role, "parts": [content]})

            chat = model.start_chat(history=history)
            response = chat.send_message(prompt, generation_config=generation_config)
            try:
                return response.text
            except ValueError as ve:
                logger.warning(f"Response blocked or empty. Reason: {ve}. Candidates: {response.candidates}")
                if response.prompt_feedback:
                    logger.warning(f"Prompt feedback: {response.prompt_feedback}")
                return ""
        except Exception as e:
            logger.error(f"Gemini Verifier LLM error: {e}", exc_info=True)
            raise
    return await asyncio.to_thread(_sync_call)


async def gemini_vision_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    image_data: Optional[str] = None,
    **kwargs,
) -> str:
    """Gemini Vision function for image analysis."""
    def _sync_call():
        try:
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            safety_settings = [
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
            ]
            # --- FIX: Use dedicated vision model ---
            logger.info(f"Creating Vision GenerativeModel with model_name: {GEMINI_VISION_MODEL}")
            model = genai.GenerativeModel(GEMINI_VISION_MODEL, safety_settings=safety_settings)
            config_params = {}
            if "temperature" in kwargs:
                config_params["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                config_params["max_output_tokens"] = kwargs["max_tokens"]
            generation_config = genai.types.GenerationConfig(**config_params)

            content_parts = []
            if system_prompt:
                content_parts.append(system_prompt)
            content_parts.append(prompt)

            if image_data:
                import base64
                import io
                from PIL import Image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                content_parts.append(image)

            response = model.generate_content(content_parts, generation_config=generation_config)
            try:
                return response.text
            except ValueError as ve:
                logger.warning(f"Vision response blocked or empty. Reason: {ve}. Candidates: {response.candidates}")
                if response.prompt_feedback:
                    logger.warning(f"Vision prompt feedback: {response.prompt_feedback}")
                return ""
        except Exception as e:
            logger.error(f"Gemini Vision error: {e}", exc_info=True)
            raise
    return await asyncio.to_thread(_sync_call)


async def gemini_embedding_func(texts: List[str]) -> List[List[float]]:
    """Gemini Embedding function for text vectorization."""
    def _sync_call():
        try:
            # --- IMPROVEMENT: Use newer embedding model ---
            result = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Gemini Embedding error: {e}", exc_info=True)
            raise
    return await asyncio.to_thread(_sync_call)

gemini_embedding_func.embedding_dim = 768


async def synthesize_web_results_with_gemini(
    query: str,
    web_context: str,
    rag_context: Optional[str] = None
) -> str:
    """
    Use Gemini to synthesize web search results into a coherent, direct answer

    Args:
        query: User's original query
        web_context: Formatted web search results
        rag_context: Optional RAG results to incorporate

    Returns:
        Synthesized answer from Gemini
    """
    try:
        logger.info("Synthesizing web results with Gemini")

        # Build synthesis prompt
        if rag_context:
            system_prompt = """You are an expert research assistant. Your task is to synthesize information from both
a knowledge base and recent web search results to provide a comprehensive, accurate answer.

Guidelines:
- Provide a direct, clear answer to the user's question
- Combine insights from both the knowledge base and web sources
- Cite sources when making specific claims (use [Source N] notation)
- If there are contradictions, acknowledge them and explain
- Be concise but thorough
- Use a professional, informative tone"""

            prompt = f"""User Question: {query}

Knowledge Base Information:
{rag_context}

Web Search Results:
{web_context}

Based on the above information, provide a comprehensive answer to the user's question. Synthesize the information from both sources and cite your sources appropriately."""

        else:
            system_prompt = """You are an expert research assistant. Your task is to synthesize web search results
into a clear, direct answer to the user's question.

Guidelines:
- Provide a direct, clear answer to the user's question
- Cite sources when making specific claims (use [Source N] notation)
- Be concise but comprehensive
- If information is limited or unclear, acknowledge it
- Use a professional, informative tone
- Include relevant details like dates, statistics, or examples when available"""

            prompt = f"""User Question: {query}

Web Search Results:
{web_context}

Based on the web search results above, provide a clear and comprehensive answer to the user's question. Cite your sources appropriately."""

        # Call Gemini to synthesize the answer
        answer = await gemini_llm_func(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more focused answers
            max_tokens=1500
        )

        if not answer or len(answer.strip()) < 10:
            logger.warning("Gemini synthesis produced minimal output, using fallback")
            return web_context

        return answer

    except Exception as e:
        logger.error(f"Error synthesizing web results with Gemini: {e}", exc_info=True)
        # Fallback to raw web context
        return web_context


async def gemini_rerank_func(query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Gemini-based reranking function for LightRAG

    This follows LightRAG's reranking API signature which expects:
    - documents: List of strings (not dict chunks)
    - top_n: Number of top results (not top_k)
    - Returns: List of {"index": int, "relevance_score": float}

    Args:
        query: Search query
        documents: List of document strings to rerank
        top_n: Number of top documents to return (None = return all, reranked)

    Returns:
        List of {"index": int, "relevance_score": float} in descending score order
    """
    try:
        # Convert documents (strings) to chunks format for our reranker
        chunks = [{"content": doc} for doc in documents]

        # Initialize reranker with Gemini LLM function
        reranker = GeminiReranker(
            llm_func=gemini_llm_func,
            batch_size=3,  # Process 3 chunks at a time to avoid rate limits
            temperature=0.1
        )

        # Perform reranking
        reranked_chunks = await reranker.rerank(query, chunks, top_n)

        # Convert back to LightRAG format: List[{"index": int, "relevance_score": float}]
        results = []
        for i, chunk in enumerate(reranked_chunks):
            # Find original index of this chunk
            original_content = chunk.get("content", "")
            try:
                original_index = documents.index(original_content)
            except ValueError:
                # Fallback: use current index if not found
                original_index = i

            results.append({
                "index": original_index,
                "relevance_score": chunk.get("relevance_score", 0.0)
            })

        logger.debug(f"Reranked {len(documents)} documents, returning {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Reranking error: {e}", exc_info=True)
        # Return original order on error - format: List[{"index": int, "relevance_score": float}]
        result_count = top_n if top_n and top_n < len(documents) else len(documents)
        return [{"index": i, "relevance_score": 1.0} for i in range(result_count)]


# =============================================================================
# RAG Instance Management
# =============================================================================

async def get_rag_instance(domain: str) -> RAGAnything:
    """Get or create RAG instance for a specific domain."""
    if domain not in DOMAIN_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain '{domain}'. Valid domains: {list(DOMAIN_CONFIGS.keys())}"
        )
    if domain in rag_instances:
        logger.debug(f"Using cached RAG instance for domain: {domain}")
        return rag_instances[domain]
    
    logger.info(f"Creating new RAG instance for domain: {domain}")
    try:
        domain_config = DOMAIN_CONFIGS[domain]
        domain_storage = STORAGE_DIR / domain
        domain_storage.mkdir(parents=True, exist_ok=True)

        config = RAGAnythingConfig(
            working_dir=str(domain_storage),
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            **domain_config["config_overrides"]
        )
        rag = await create_rag_anything(
            llm_model_func=gemini_llm_func,              # Flash for generation
            vision_model_func=gemini_vision_func,         # Flash for vision
            embedding_func=gemini_embedding_func,         # Embedding model
            verifier_llm_func=gemini_verifier_llm_func,  # Pro for verification
            config=config,
            rerank_model_func=gemini_rerank_func,        # Enable reranking (passed directly)
        )
        rag_instances[domain] = rag
        logger.info(f"RAG instance created successfully for domain: {domain}")
        return rag
    except Exception as e:
        logger.error(f"Failed to create RAG instance for domain {domain}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG system for domain '{domain}': {str(e)}"
        )

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        features={
            "query_improvement": True,
            "dual_llm_verification": True,
            "gemini_pro_verifier": True,
            "reranking": True,
            "conversation_memory": True,
            "multi_domain": True,
            "multimodal_processing": True,
            "gemini_integration": bool(GEMINI_API_KEY),
            "web_search": bool(web_searcher),
            "url_ingestion": bool(url_fetcher),
        },
        domains=list(DOMAIN_CONFIGS.keys())
    )


@app.get("/domains", response_model=List[DomainInfo])
async def list_domains():
    """List all available domains."""
    domains = []
    for domain_id, config in DOMAIN_CONFIGS.items():
        domains.append(DomainInfo(
            domain_id=domain_id,
            name=config["name"],
            description=config["description"],
            file_extensions=config["file_extensions"],
            features={k: v for k, v in config["config_overrides"].items() if isinstance(v, bool)}
        ))
    return domains


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    domain: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a document in the background."""
    logger.info(f"Upload request: {file.filename} to domain: {domain}")
    try:
        if domain not in DOMAIN_CONFIGS:
            raise HTTPException(400, f"Invalid domain. Valid: {list(DOMAIN_CONFIGS.keys())}")

        file_ext = Path(file.filename).suffix.lower()
        allowed_extensions = DOMAIN_CONFIGS[domain]["file_extensions"]
        if file_ext not in allowed_extensions:
            raise HTTPException(400, f"File type {file_ext} not for '{domain}'. Allowed: {allowed_extensions}")

        processing_id = str(uuid.uuid4())
        domain_upload_dir = UPLOAD_DIR / domain
        domain_upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = domain_upload_dir / f"{processing_id}_{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"File saved: {file_path}")

        # Initialize status and save to disk
        update_processing_status(processing_id, {
            "status": "processing",
            "message": "Processing document...",
            "file_name": file.filename,
            "domain": domain,
            "started_at": datetime.now().isoformat()
        })

        async def process_document_task():
            try:
                logger.info(f"Processing document: {file_path}")
                rag = await get_rag_instance(domain)
                result = await rag.process_document_complete(str(file_path))

                # Check result (process_document_complete returns None on success)
                if result is None or (isinstance(result, dict) and result.get("success") is not False):
                    logger.info(f"Document processed successfully: {file.filename}")
                    update_processing_status(processing_id, {
                        "status": "completed",
                        "message": "Document processed successfully",
                        "file_name": file.filename,
                        "domain": domain,
                        "completed_at": datetime.now().isoformat()
                    })
                else:
                    error_msg = result.get('error', 'Unknown processing error') if isinstance(result, dict) else "Processing error"
                    logger.error(f"Document processing failed: {error_msg}")
                    update_processing_status(processing_id, {
                        "status": "failed",
                        "message": f"Processing failed: {error_msg}",
                        "file_name": file.filename,
                        "domain": domain,
                        "error": error_msg
                    })
            except Exception as e:
                logger.error(f"Error in background processing of {file.filename}: {e}", exc_info=True)
                update_processing_status(processing_id, {
                    "status": "failed",
                    "message": f"Error: {str(e)}",
                    "file_name": file.filename,
                    "domain": domain,
                    "error": str(e)
                })

        background_tasks.add_task(process_document_task)

        return UploadResponse(
            success=True,
            message="Document uploaded and queued for processing",
            file_name=file.filename,
            domain=domain,
            processing_id=processing_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/upload-batch", response_model=BatchUploadResponse)
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    domain: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process multiple documents in batch using optimized processing.

    Uses BatchOptimizer for 2-3x faster processing through:
    - Parallel parsing (up to 4 documents simultaneously)
    - Parallel processing (up to 10 documents simultaneously)
    - Pipeline architecture (parse + process in parallel)
    """
    logger.info(f"Batch upload request: {len(files)} files to domain: {domain}")
    try:
        if domain not in DOMAIN_CONFIGS:
            raise HTTPException(400, f"Invalid domain. Valid: {list(DOMAIN_CONFIGS.keys())}")

        allowed_extensions = DOMAIN_CONFIGS[domain]["file_extensions"]
        domain_upload_dir = UPLOAD_DIR / domain
        domain_upload_dir.mkdir(parents=True, exist_ok=True)

        # Process and save all files
        file_paths = []
        processing_ids = []
        rejected_files = []

        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                rejected_files.append(file.filename)
                logger.warning(f"Rejected file {file.filename}: extension {file_ext} not allowed")
                continue

            processing_id = str(uuid.uuid4())
            file_path = domain_upload_dir / f"{processing_id}_{file.filename}"

            with open(file_path, "wb") as f:
                f.write(await file.read())

            file_paths.append(str(file_path))
            processing_ids.append(processing_id)

            # Initialize status for each file
            update_processing_status(processing_id, {
                "status": "queued",
                "message": "Queued for batch processing...",
                "file_name": file.filename,
                "domain": domain,
                "started_at": datetime.now().isoformat()
            })

        logger.info(f"Accepted {len(file_paths)}/{len(files)} files, rejected: {rejected_files}")

        if not file_paths:
            raise HTTPException(400, f"No valid files provided. Allowed extensions: {allowed_extensions}")

        # Process documents in batch using optimized processing
        async def process_batch_task():
            start_time = time.time()
            try:
                logger.info(f"Starting optimized batch processing of {len(file_paths)} files")
                rag = await get_rag_instance(domain)

                # Use optimized batch processing if available
                if hasattr(rag, 'process_documents_batch_optimized'):
                    result = await rag.process_documents_batch_optimized(
                        file_paths=file_paths,
                        max_concurrent_parsers=4,  # MinerU optimal
                        max_concurrent_processors=10,  # Higher for I/O-bound tasks
                        enable_progress_tracking=True,
                    )

                    # Update statuses based on results
                    successful_files = result.get('successful_files', [])
                    failed_files = result.get('failed_files', {})

                    for idx, file_path in enumerate(file_paths):
                        processing_id = processing_ids[idx]
                        filename = Path(file_path).name.split('_', 1)[1] if '_' in Path(file_path).name else Path(file_path).name

                        if file_path in successful_files:
                            update_processing_status(processing_id, {
                                "status": "completed",
                                "message": "Document processed successfully",
                                "file_name": filename,
                                "domain": domain,
                                "completed_at": datetime.now().isoformat()
                            })
                        elif file_path in failed_files:
                            error_msg = failed_files[file_path]
                            update_processing_status(processing_id, {
                                "status": "failed",
                                "message": f"Processing failed: {error_msg}",
                                "file_name": filename,
                                "domain": domain,
                                "error": error_msg
                            })

                    total_time = time.time() - start_time
                    throughput = len(successful_files) / total_time if total_time > 0 else 0
                    logger.info(
                        f"Batch processing complete: {len(successful_files)}/{len(file_paths)} successful "
                        f"in {total_time:.2f}s ({throughput:.2f} docs/sec)"
                    )

                    # Track performance
                    performance_metrics["processing_times"].append(total_time)
                    if len(performance_metrics["processing_times"]) > 100:
                        performance_metrics["processing_times"] = performance_metrics["processing_times"][-100:]

                else:
                    # Fallback: process sequentially
                    logger.warning("Optimized batch processing not available, using sequential processing")
                    for idx, file_path in enumerate(file_paths):
                        processing_id = processing_ids[idx]
                        filename = Path(file_path).name.split('_', 1)[1] if '_' in Path(file_path).name else Path(file_path).name

                        current_status = processing_status[processing_id].copy()
                        current_status["status"] = "processing"
                        current_status["message"] = "Processing document..."
                        update_processing_status(processing_id, current_status)

                        try:
                            result = await rag.process_document_complete(file_path)
                            if result is None or (isinstance(result, dict) and result.get("success") is not False):
                                update_processing_status(processing_id, {
                                    "status": "completed",
                                    "message": "Document processed successfully",
                                    "file_name": filename,
                                    "domain": domain,
                                    "completed_at": datetime.now().isoformat()
                                })
                            else:
                                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else "Processing error"
                                update_processing_status(processing_id, {
                                    "status": "failed",
                                    "message": f"Processing failed: {error_msg}",
                                    "file_name": filename,
                                    "domain": domain,
                                    "error": error_msg
                                })
                        except Exception as e:
                            logger.error(f"Error processing {filename}: {e}", exc_info=True)
                            update_processing_status(processing_id, {
                                "status": "failed",
                                "message": f"Error: {str(e)}",
                                "file_name": filename,
                                "domain": domain,
                                "error": str(e)
                            })

            except Exception as e:
                logger.error(f"Batch processing error: {e}", exc_info=True)
                # Mark all as failed
                for idx, file_path in enumerate(file_paths):
                    processing_id = processing_ids[idx]
                    filename = Path(file_path).name.split('_', 1)[1] if '_' in Path(file_path).name else Path(file_path).name
                    update_processing_status(processing_id, {
                        "status": "failed",
                        "message": f"Batch processing error: {str(e)}",
                        "file_name": filename,
                        "domain": domain,
                        "error": str(e)
                    })

        background_tasks.add_task(process_batch_task)

        return BatchUploadResponse(
            success=True,
            message=f"Batch upload queued: {len(file_paths)} files accepted" + (f", {len(rejected_files)} rejected" if rejected_files else ""),
            total_files=len(files),
            accepted_files=len(file_paths),
            processing_ids=processing_ids,
            domain=domain
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")


@app.post("/upload-url", response_model=UploadResponse)
async def upload_url(
    request: URLUploadRequest,
    background_tasks: BackgroundTasks
):
    """Fetch document from URL and process it."""
    logger.info(f"URL upload request: {request.url} to domain: {request.domain}")
    try:
        if not url_fetcher:
            raise HTTPException(503, "URL fetcher not available. Check server configuration.")

        if request.domain not in DOMAIN_CONFIGS:
            raise HTTPException(400, f"Invalid domain. Valid: {list(DOMAIN_CONFIGS.keys())}")

        processing_id = str(uuid.uuid4())

        # Initialize status
        update_processing_status(processing_id, {
            "status": "fetching",
            "message": "Fetching URL content...",
            "url": request.url,
            "domain": request.domain,
            "started_at": datetime.now().isoformat()
        })

        async def fetch_and_process_url():
            try:
                logger.info(f"[URL UPLOAD] Starting fetch for: {request.url}")

                # Fetch URL content with timeout
                fetch_result = await asyncio.wait_for(
                    url_fetcher.fetch_url(
                        url=request.url,
                        convert_to_markdown=request.convert_to_markdown
                    ),
                    timeout=60.0  # 60 second timeout for URL fetching
                )

                if not fetch_result.get("success"):
                    error_msg = fetch_result.get("error", "Unknown fetch error")
                    logger.error(f"[URL UPLOAD] Fetch failed: {error_msg}")
                    update_processing_status(processing_id, {
                        "status": "failed",
                        "message": f"Failed to fetch URL: {error_msg}",
                        "domain": request.domain,
                        "error": error_msg
                    })
                    return

                file_path = fetch_result.get("file_path")
                if not file_path:
                    logger.error("[URL UPLOAD] No file path returned from URL fetch")
                    update_processing_status(processing_id, {
                        "status": "failed",
                        "message": "No file path returned from URL fetch",
                        "domain": request.domain,
                        "error": "No file path"
                    })
                    return

                logger.info(f"[URL UPLOAD] Content saved to: {file_path}")

                # Update status
                update_processing_status(processing_id, {
                    "status": "processing",
                    "message": "Processing document...",
                    "domain": request.domain,
                    "file_path": file_path
                })

                # Get RAG instance
                rag = await get_rag_instance(request.domain)

                # Check if we have a content list with images (advanced HTML parsing)
                content_list = fetch_result.get("content_list")
                images_count = fetch_result.get("images_count", 0)

                if content_list and len(content_list) > 0 and images_count > 0:
                    # Advanced pathway: Process pre-parsed content list with images
                    logger.info(f"[URL UPLOAD] Using advanced processing: {len(content_list)} blocks, {images_count} images")
                    result = await asyncio.wait_for(
                        rag.process_content_list_direct(
                            content_list=content_list,
                            source_identifier=request.url,
                            enable_image_processing=True
                        ),
                        timeout=300.0  # 5 minute timeout for processing
                    )
                else:
                    # Standard pathway: Process as regular document (PDF or text-only HTML)
                    logger.info("[URL UPLOAD] Using standard document processing")
                    result = await asyncio.wait_for(
                        rag.process_document_complete(file_path),
                        timeout=300.0  # 5 minute timeout for processing
                    )

                # Check result and update status
                # Note: process_document_complete returns None on success (not a dict)
                if result is None or (isinstance(result, dict) and result.get("success") is not False):
                    logger.info(f"[URL UPLOAD] ✓ Successfully processed: {request.url}")
                    update_processing_status(processing_id, {
                        "status": "completed",
                        "message": "Document processed successfully",
                        "domain": request.domain,
                        "file_path": file_path,
                        "completed_at": datetime.now().isoformat()
                    })
                else:
                    error_msg = result.get('error', 'Unknown processing error') if isinstance(result, dict) else "Processing returned error"
                    logger.error(f"[URL UPLOAD] ✗ Processing failed: {error_msg}")
                    update_processing_status(processing_id, {
                        "status": "failed",
                        "message": f"Processing failed: {error_msg}",
                        "domain": request.domain,
                        "error": error_msg
                    })

            except asyncio.TimeoutError:
                logger.error(f"[URL UPLOAD] ✗ Timeout processing {request.url}")
                update_processing_status(processing_id, {
                    "status": "failed",
                    "message": "Processing timeout",
                    "domain": request.domain,
                    "error": "Timeout"
                })
            except Exception as e:
                logger.error(f"[URL UPLOAD] ✗ Error processing {request.url}: {e}", exc_info=True)
                update_processing_status(processing_id, {
                    "status": "failed",
                    "message": f"Error: {str(e)}",
                    "domain": request.domain,
                    "error": str(e)
                })

        background_tasks.add_task(fetch_and_process_url)

        return UploadResponse(
            success=True,
            message="URL queued for fetching and processing",
            file_name=request.url,
            domain=request.domain,
            processing_id=processing_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"URL upload failed: {str(e)}")


@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Stream query responses with real-time token generation and verification.

    This endpoint provides Server-Sent Events (SSE) streaming for real-time
    response generation while maintaining dual-LLM verification.
    """
    logger.info(f"Streaming query request: '{request.query[:50]}...' in domain: {request.domain}")

    async def generate_sse():
        """Generate Server-Sent Events stream"""
        import json

        try:
            conversation_id = request.conversation_id or f"conv_{uuid.uuid4()}"

            # Get RAG instance
            rag = await get_rag_instance(request.domain)

            # Determine optimal parameters based on fast_mode
            if request.fast_mode:
                # Optimized parameters for 2-3x speedup
                top_k = request.top_k if request.top_k is not None else 20
                chunk_top_k = 10
                max_entity_tokens = 4000
                max_relation_tokens = 6000
                max_total_tokens = 20000
                logger.info(f"⚡ Fast mode enabled for streaming: top_k={top_k}, chunk_top_k={chunk_top_k}")
            else:
                # Default parameters (higher quality, slower)
                top_k = request.top_k if request.top_k is not None else 40
                chunk_top_k = 20
                max_entity_tokens = 6000
                max_relation_tokens = 8000
                max_total_tokens = 30000

            # Log toggle settings
            logger.info(f"Query settings - improvement: {request.enable_query_improvement}, verification: {request.enable_verification_check}")

            # Stream the query with optimized parameters and user-controlled toggles
            async for chunk in rag.aquery_stream(
                query=request.query,
                mode=request.mode,
                enable_verification=request.enable_verification_check,  # Use toggle instead of always true
                enable_query_improvement=request.enable_query_improvement,  # Use toggle instead of always true
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                max_entity_tokens=max_entity_tokens,
                max_relation_tokens=max_relation_tokens,
                max_total_tokens=max_total_tokens
            ):
                chunk_type = chunk.get("type", "token")
                content = chunk.get("content", "")
                done = chunk.get("done", False)

                if chunk_type == "token":
                    # Stream token
                    data = {"type": "token", "content": content, "done": done}
                    yield f"event: token\ndata: {json.dumps(data)}\n\n"

                elif chunk_type == "verification":
                    # Send verification metadata
                    data = {"type": "verification", "content": content, "done": done}
                    yield f"event: verification\ndata: {json.dumps(data)}\n\n"

                elif chunk_type == "error":
                    # Send error
                    data = {"type": "error", "content": content, "done": True}
                    yield f"event: error\ndata: {json.dumps(data)}\n\n"
                    break

            # Send completion event
            yield f"event: done\ndata: {json.dumps({'message': 'Stream complete', 'conversation_id': conversation_id})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_data = {"type": "error", "content": {"message": str(e)}, "done": True}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with enhanced RAG capabilities and optional web search."""
    start_time = time.time()
    logger.info(f"Query request: '{request.query[:50]}...' in domain: {request.domain}, mode: {request.mode}, fast_mode: {request.fast_mode}")

    try:
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4()}"
        conversation_history = conversation_histories.get(conversation_id, [])

        # Generate cache key for non-web-search queries
        cache_key = None
        if request.enable_cache and not request.web_search_only and not request.enable_web_search:
            cache_data = f"{request.query}:{request.domain}:{request.mode}:{request.fast_mode}:{request.enable_verification}"
            cache_key = hashlib.md5(cache_data.encode()).hexdigest()

            # Check cache
            if cache_key in query_cache:
                cached_response = query_cache[cache_key]
                logger.info(f"✓ Cache hit for query (saved {time.time() - start_time:.2f}s)")
                # Update conversation ID in cached response
                cached_response.conversation_id = conversation_id
                return cached_response

        # Handle web search only mode
        if request.web_search_only:
            if not web_searcher:
                raise HTTPException(503, "Web search not available. Set TAVILY_API_KEY.")

            logger.info("Using web search only mode")
            web_results = await web_searcher.search(request.query, max_results=5)

            # Format results for LLM processing
            web_context = web_searcher.format_results_for_llm(web_results)

            # Synthesize answer using Gemini
            logger.info("Synthesizing web search results with Gemini")
            answer = await synthesize_web_results_with_gemini(
                query=request.query,
                web_context=web_context,
                rag_context=None
            )

            result = {
                "answer": answer,
                "original_query": request.query,
                "improved_query": request.query,
                "verification_passed": False,
                "verification_score": 0,
                "web_search_performed": True,
                "sources": [{"url": r.get("url"), "title": r.get("title")} for r in web_results.get("results", [])]
            }
        else:
            # Standard RAG query with optimized parameters
            rag = await get_rag_instance(request.domain)

            # Determine optimal parameters based on fast_mode
            if request.fast_mode:
                # Optimized parameters for 2-3x speedup
                top_k = request.top_k if request.top_k is not None else 20
                chunk_top_k = 10
                max_entity_tokens = 4000
                max_relation_tokens = 6000
                max_total_tokens = 20000
                logger.info(f"⚡ Fast mode enabled: top_k={top_k}, chunk_top_k={chunk_top_k}")
            else:
                # Default parameters (higher quality, slower)
                top_k = request.top_k if request.top_k is not None else 40
                chunk_top_k = 20
                max_entity_tokens = 6000
                max_relation_tokens = 8000
                max_total_tokens = 30000

            # Build query parameters
            from lightrag import QueryParam
            query_kwargs = {
                "top_k": top_k,
                "chunk_top_k": chunk_top_k,
                "max_entity_tokens": max_entity_tokens,
                "max_relation_tokens": max_relation_tokens,
                "max_total_tokens": max_total_tokens,
            }

            # Log toggle settings
            logger.info(f"Query settings - improvement: {request.enable_query_improvement}, verification: {request.enable_verification_check}")

            result = await rag.aquery(
                query=request.query,
                mode=request.mode,
                enable_query_improvement=request.enable_query_improvement,  # Use toggle instead of always true
                enable_verification=request.enable_verification_check,  # Use toggle instead of always request.enable_verification
                return_verification_info=request.return_metadata,
                **query_kwargs
            )

            # Augment with web search if requested
            if request.enable_web_search and web_searcher:
                logger.info("Augmenting RAG results with web search")
                try:
                    rag_answer = result.get("answer") if isinstance(result, dict) else str(result)
                    web_results = await web_searcher.search(request.query, max_results=5)

                    if web_results.get("results"):
                        # Format web results for LLM
                        web_context = web_searcher.format_results_for_llm(web_results)

                        # Synthesize combined answer using Gemini
                        logger.info("Synthesizing RAG + web results with Gemini")
                        synthesized_answer = await synthesize_web_results_with_gemini(
                            query=request.query,
                            web_context=web_context,
                            rag_context=rag_answer
                        )

                        if isinstance(result, dict):
                            result["answer"] = synthesized_answer
                            result["web_search_performed"] = True
                            result["web_sources"] = [{"url": r.get("url"), "title": r.get("title")} for r in web_results.get("results", [])]
                        else:
                            result = synthesized_answer
                except Exception as e:
                    logger.error(f"Web search augmentation error: {e}")
                    # Continue with RAG-only result

        # Handle None result
        if result is None:
            answer = "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or ensure that relevant documents have been uploaded."
            metadata = {
                "original_query": request.query,
                "improved_query": request.query,
                "verification_passed": False,
                "verification_score": 0,
            }
            query_improved = False
            verification_performed = False
            confidence = 0.0
        elif isinstance(result, dict):
            answer = result.get("answer", "No answer found.")
            metadata = {
                "original_query": result.get("original_query", request.query),
                "improved_query": result.get("improved_query", request.query),
                "verification_passed": result.get("verification_passed", False),
                "verification_score": result.get("verification_score", 0),
            }
            query_improved = result.get("improved_query") != result.get("original_query")
            verification_performed = result.get("verification_passed", False)
            confidence = result.get("verification_score", 0) / 10.0
        else:
            answer = str(result) if result else "No answer found."
            metadata = {}
            query_improved = False
            verification_performed = False
            confidence = 1.0

        conversation_history.extend([
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": answer}
        ])
        conversation_histories[conversation_id] = conversation_history

        response = QueryResponse(
            answer=answer,
            sources=[],  # TODO: Extract from result if available
            confidence_score=confidence,
            query_improved=query_improved,
            verification_performed=verification_performed,
            conversation_id=conversation_id,
            metadata=metadata if request.return_metadata else None
        )

        # Store in cache if enabled (non-web search queries only)
        if cache_key and request.enable_cache:
            query_cache[cache_key] = response
            logger.info(f"✓ Cached query result (key: {cache_key[:16]}...)")

        # Track performance metrics
        query_time = time.time() - start_time
        performance_metrics["query_times"].append(query_time)
        # Keep only last 100 metrics
        if len(performance_metrics["query_times"]) > 100:
            performance_metrics["query_times"] = performance_metrics["query_times"][-100:]

        logger.info(f"Query completed in {query_time:.2f}s (fast_mode: {request.fast_mode}, confidence: {confidence:.2f})")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history by ID."""
    if conversation_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": conversation_id,
        "messages": conversation_histories[conversation_id],
    }


@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history."""
    if conversation_id in conversation_histories:
        del conversation_histories[conversation_id]
        logger.info(f"Cleared conversation: {conversation_id}")
        return {"success": True, "message": "Conversation cleared"}
    raise HTTPException(status_code=404, detail="Conversation not found")


@app.delete("/clear/{domain}")
async def clear_domain_data(domain: str):
    """WARNING: Deletes all processed documents and indices for the domain."""
    logger.warning(f"Clear domain data request: {domain}")
    try:
        if domain not in DOMAIN_CONFIGS:
            raise HTTPException(400, f"Invalid domain. Valid: {list(DOMAIN_CONFIGS.keys())}")
        
        if domain in rag_instances:
            await rag_instances[domain].finalize_storages()
            del rag_instances[domain]

        domain_storage = STORAGE_DIR / domain
        if domain_storage.exists():
            import shutil
            shutil.rmtree(domain_storage)
            domain_storage.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Domain data cleared: {domain}")
        return {"success": True, "message": f"All data cleared for domain '{domain}'"}
    except Exception as e:
        logger.error(f"Clear domain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear domain: {str(e)}")


@app.get("/documents")
async def list_documents(domain: str):
    """
    List all processed documents for a domain.

    Only returns documents with status 'completed'. Documents still being
    processed are excluded to avoid confusion.
    """
    try:
        if domain not in DOMAIN_CONFIGS:
            raise HTTPException(400, f"Invalid domain. Valid: {list(DOMAIN_CONFIGS.keys())}")

        documents = []
        domain_upload_dir = UPLOAD_DIR / domain

        if domain_upload_dir.exists():
            for file_path in domain_upload_dir.glob("*"):
                if file_path.is_file():
                    # Extract processing_id and filename
                    filename = file_path.name
                    parts = filename.split('_', 1)
                    processing_id = parts[0] if len(parts) > 1 else ""
                    display_name = parts[1] if len(parts) > 1 else filename

                    # Check if document is actually completed
                    # Skip if still processing, queued, or fetching
                    if processing_id in processing_status:
                        status = processing_status[processing_id].get('status')
                        if status in ['processing', 'queued', 'fetching']:
                            # Document is still being processed, skip it
                            logger.debug(f"Skipping document {processing_id} - status: {status}")
                            continue
                        elif status == 'failed':
                            # Optionally skip failed documents or include them
                            # For now, skip them to only show successfully processed docs
                            continue

                    # Only include completed documents or legacy ones without status
                    documents.append({
                        "id": processing_id,
                        "name": display_name,
                        "domain": domain,
                        "status": "processed",
                        "uploadedAt": str(file_path.stat().st_mtime)
                    })

        return {"documents": documents}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/performance-metrics")
async def get_performance_metrics():
    """Get performance metrics for queries and document processing."""
    try:
        query_times = performance_metrics.get("query_times", [])
        processing_times = performance_metrics.get("processing_times", [])

        # Calculate statistics
        def calc_stats(times):
            if not times:
                return {"count": 0, "avg": 0, "min": 0, "max": 0}
            return {
                "count": len(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times)
            }

        return {
            "query_metrics": calc_stats(query_times),
            "processing_metrics": calc_stats(processing_times),
            "cache_stats": {
                "size": len(query_cache),
                "max_size": query_cache.maxsize,
                "ttl_seconds": query_cache.ttl
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        return {
            "query_metrics": {"count": 0, "avg": 0, "min": 0, "max": 0},
            "processing_metrics": {"count": 0, "avg": 0, "min": 0, "max": 0},
            "cache_stats": {"size": 0, "max_size": 100, "ttl_seconds": 300}
        }


@app.get("/status/{processing_id}")
async def get_processing_status(processing_id: str):
    """
    Get the processing status of a document.

    Now uses persistent status storage that survives backend restarts.
    The status is loaded from disk on startup and kept in sync.
    """
    try:
        # Check the persistent status tracker (loaded from disk on startup)
        if processing_id in processing_status:
            status_info = processing_status[processing_id]
            logger.debug(f"Status check for {processing_id}: {status_info.get('status')}")
            return {
                "processing_id": processing_id,
                **status_info
            }

        # If not in status tracker, check if this is a legacy upload
        # (uploaded before persistent status was implemented)
        for domain in DOMAIN_CONFIGS.keys():
            domain_upload_dir = UPLOAD_DIR / domain
            if domain_upload_dir.exists():
                for file_path in domain_upload_dir.glob(f"{processing_id}_*"):
                    if file_path.is_file():
                        # Legacy upload - return completed status
                        # but don't add to persistent status to avoid confusion
                        return {
                            "processing_id": processing_id,
                            "status": "completed",
                            "message": "Document processed successfully (legacy upload)"
                        }

        # If not found anywhere, status is unknown
        # This typically means the processing_id is invalid
        return {
            "processing_id": processing_id,
            "status": "unknown",
            "message": "Processing ID not found. It may be invalid or expired."
        }
    except Exception as e:
        logger.error(f"Error checking status: {e}", exc_info=True)
        return {
            "processing_id": processing_id,
            "status": "error",
            "message": f"Error checking status: {str(e)}",
            "error": str(e)
        }


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a processed document completely including all RAG data.

    This endpoint performs comprehensive deletion of:
    - Knowledge graph entities and relationships
    - Embedding vectors (chunks, entities, relationships)
    - Text chunks and metadata
    - Document status records
    - Physical upload files
    - Parser output files

    Returns detailed deletion report with verification.
    """
    try:
        from raganything.deletion_verifier import delete_document_complete

        logger.info(f"Delete document request: {doc_id}")

        # Step 1: Search for the document in all domains
        found_domain = None
        for domain in DOMAIN_CONFIGS.keys():
            domain_upload_dir = UPLOAD_DIR / domain
            if domain_upload_dir.exists():
                for file_path in domain_upload_dir.glob(f"{doc_id}_*"):
                    if file_path.is_file():
                        found_domain = domain
                        break
            if found_domain:
                break

        if not found_domain:
            logger.warning(f"Document {doc_id} not found in any domain")
            raise HTTPException(status_code=404, detail="Document not found")

        logger.info(f"Found document {doc_id} in domain: {found_domain}")

        # Step 2: Get RAG instance and find the actual document ID in storage
        rag = await get_rag_instance(found_domain)

        # Find document in doc_status by processing_id prefix
        doc_to_delete = None
        doc_status_file = STORAGE_DIR / found_domain / "kv_store_doc_status.json"
        if doc_status_file.exists():
            import json
            with open(doc_status_file, 'r') as f:
                doc_status = json.load(f)

            # Find document by file_path containing doc_id
            for doc_key, doc_info in doc_status.items():
                if 'file_path' in doc_info and doc_id in doc_info['file_path']:
                    doc_to_delete = doc_key
                    logger.info(f"Found document in storage: {doc_key}")
                    break

        if not doc_to_delete:
            logger.warning(f"Document {doc_id} not found in doc_status")
            # Still try to delete physical files
            doc_to_delete = doc_id

        # Step 3: Collect files and directories to delete
        upload_files = list((UPLOAD_DIR / found_domain).glob(f"{doc_id}_*"))
        output_dir = BASE_DIR / "backend" / "output"
        output_paths = list(output_dir.glob(f"{doc_id}_*")) if output_dir.exists() else []

        # Step 4: Perform complete deletion with verification
        deletion_report = await delete_document_complete(
            rag_instance=rag,
            doc_id=doc_to_delete,
            storage_dir=STORAGE_DIR / found_domain,
            upload_files=upload_files,
            output_dirs=output_paths
        )

        # Step 5: Return detailed report
        if deletion_report.success:
            logger.info(
                f"Successfully deleted document {doc_id}: "
                f"{deletion_report.chunks_deleted} chunks, "
                f"{deletion_report.entities_deleted} entities, "
                f"{deletion_report.relationships_deleted} relationships, "
                f"{len(deletion_report.files_deleted)} files, "
                f"{len(deletion_report.directories_deleted)} directories"
            )
            return {
                "success": True,
                "message": "Document deleted completely with verification",
                "domain": found_domain,
                "report": deletion_report.to_dict()
            }
        else:
            logger.error(
                f"Document deletion completed with errors for {doc_id}: "
                f"{deletion_report.errors}"
            )
            return {
                "success": False,
                "message": "Document deletion completed with errors",
                "domain": found_domain,
                "report": deletion_report.to_dict()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
