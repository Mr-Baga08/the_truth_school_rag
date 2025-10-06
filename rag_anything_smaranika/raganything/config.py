"""
Configuration classes for RAGAnything

Contains configuration dataclasses with environment variable support
"""

from dataclasses import dataclass, field
from typing import List
from lightrag.utils import get_env_value


@dataclass
class RAGAnythingConfig:
    """Configuration class for RAGAnything with environment variable support"""

    # Directory Configuration
    # ---
    working_dir: str = field(default=get_env_value("WORKING_DIR", "./rag_storage", str))
    """Directory where RAG storage and cache files are stored."""

    # Parser Configuration
    # ---
    parse_method: str = field(default=get_env_value("PARSE_METHOD", "auto", str))
    """Default parsing method for document parsing: 'auto', 'ocr', or 'txt'."""

    parser_output_dir: str = field(default=get_env_value("OUTPUT_DIR", "./output", str))
    """Default output directory for parsed content."""

    parser: str = field(default=get_env_value("PARSER", "mineru", str))
    """Parser selection: 'mineru' or 'docling'."""

    display_content_stats: bool = field(
        default=get_env_value("DISPLAY_CONTENT_STATS", True, bool)
    )
    """Whether to display content statistics during parsing."""

    # Multimodal Processing Configuration
    # ---
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool)
    )
    """Enable image content processing."""

    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", True, bool)
    )
    """Enable table content processing."""

    enable_equation_processing: bool = field(
        default=get_env_value("ENABLE_EQUATION_PROCESSING", True, bool)
    )
    """Enable equation content processing."""

    # Batch Processing Configuration
    # ---
    max_concurrent_files: int = field(
        default=get_env_value("MAX_CONCURRENT_FILES", 1, int)
    )
    """Maximum number of files to process concurrently."""

    supported_file_extensions: List[str] = field(
        default_factory=lambda: get_env_value(
            "SUPPORTED_FILE_EXTENSIONS",
            ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md",
            str,
        ).split(",")
    )
    """List of supported file extensions for batch processing."""

    recursive_folder_processing: bool = field(
        default=get_env_value("RECURSIVE_FOLDER_PROCESSING", True, bool)
    )
    """Whether to recursively process subfolders in batch mode."""

    # Context Extraction Configuration
    # ---
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", 1, int))
    """Number of pages/chunks to include before and after current item for context."""

    context_mode: str = field(default=get_env_value("CONTEXT_MODE", "page", str))
    """Context extraction mode: 'page' for page-based, 'chunk' for chunk-based."""

    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", 2000, int)
    )
    """Maximum number of tokens in extracted context."""

    include_headers: bool = field(default=get_env_value("INCLUDE_HEADERS", True, bool))
    """Whether to include document headers and titles in context."""

    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", True, bool)
    )
    """Whether to include image/table captions in context."""

    context_filter_content_types: List[str] = field(
        default_factory=lambda: get_env_value(
            "CONTEXT_FILTER_CONTENT_TYPES", "text", str
        ).split(",")
    )
    """Content types to include in context extraction (e.g., 'text', 'image', 'table')."""

    content_format: str = field(default=get_env_value("CONTENT_FORMAT", "minerU", str))
    """Default content format for context extraction when processing documents."""


    # Query Improvement Configuration
    # ---
    enable_query_improvement: bool = field(
        default=get_env_value("ENABLE_QUERY_IMPROVEMENT", False, bool)
    )
    """Enable query improvement before retrieval."""

    query_improvement_method: str = field(
        default=get_env_value("QUERY_IMPROVEMENT_METHOD", "rewrite", str)
    )
    """Query improvement method: 'rewrite', 'expand', 'decompose', 'llm', 'rules', or 'hybrid'."""

    # NEW: Additional Query Improvement Settings
    enable_query_rewriting: bool = field(
        default=get_env_value("ENABLE_QUERY_REWRITING", False, bool)
    )
    """Enable query rewriting with conversation context for better understanding."""

    extract_query_entities: bool = field(
        default=get_env_value("EXTRACT_QUERY_ENTITIES", False, bool)
    )
    """Extract named entities from queries for enhanced retrieval targeting."""

    expand_abbreviations: bool = field(
        default=get_env_value("EXPAND_ABBREVIATIONS", False, bool)
    )
    """Expand domain-specific abbreviations in queries (e.g., 'BP' -> 'blood pressure')."""

    add_domain_keywords: bool = field(
        default=get_env_value("ADD_DOMAIN_KEYWORDS", False, bool)
    )
    """Add relevant domain-specific keywords to queries for better coverage."""

    # Dual-LLM Verification Configuration
    # ---
    enable_dual_llm_verification: bool = field(
        default=get_env_value("ENABLE_DUAL_LLM_VERIFICATION", False, bool)
    )
    """Enable two-LLM generation and verification system."""

    enable_answer_verification: bool = field(
        default=get_env_value("ENABLE_ANSWER_VERIFICATION", False, bool)
    )
    """Enable answer verification with multi-criteria evaluation."""

    verifier_model_name: str = field(
        default=get_env_value("VERIFIER_MODEL_NAME", "gpt-4o", str)
    )
    """Model name for the verifier LLM (typically more powerful than generator)."""

    verification_llm_model: str = field(
        default=get_env_value("VERIFICATION_LLM_MODEL", "gpt-4o", str)
    )
    """Alternative field name for verification model (alias for verifier_model_name)."""

    enable_answer_modification: bool = field(
        default=get_env_value("ENABLE_ANSWER_MODIFICATION", False, bool)
    )
    """Enable answer modification based on verification feedback."""

    verification_threshold: float = field(
        default=get_env_value("VERIFICATION_THRESHOLD", 7.0, float)
    )
    """Minimum verification score (out of 10) to accept answer without modification."""

    min_confidence_threshold: float = field(
        default=get_env_value("MIN_CONFIDENCE_THRESHOLD", 0.7, float)
    )
    """Minimum confidence score (0-1) required to accept verification results."""

    max_verification_retries: int = field(
        default=get_env_value("MAX_VERIFICATION_RETRIES", 2, int)
    )
    """Maximum number of answer modification attempts based on verification."""

    max_verification_iterations: int = field(
        default=get_env_value("MAX_VERIFICATION_ITERATIONS", 2, int)
    )
    """Alternative field name for max retries (alias for max_verification_retries)."""

    # NEW: Additional Verification Criteria
    check_factual_consistency: bool = field(
        default=get_env_value("CHECK_FACTUAL_CONSISTENCY", True, bool)
    )
    """Check if answer is factually consistent with retrieved context."""

    check_completeness: bool = field(
        default=get_env_value("CHECK_COMPLETENESS", True, bool)
    )
    """Check if answer addresses all aspects of the query."""

    check_relevance: bool = field(
        default=get_env_value("CHECK_RELEVANCE", True, bool)
    )
    """Check if answer is relevant to the original query."""

    add_confidence_score: bool = field(
        default=get_env_value("ADD_CONFIDENCE_SCORE", False, bool)
    )
    """Add confidence score to verification results."""

    # Domain & Memory Configuration
    # ---
    domain: str = field(
        default=get_env_value("DOMAIN", "general", str)
    )
    """Domain for domain-specific processing: 'general', 'medical', 'legal', 'financial', 'technical', 'academic'."""

    enable_domain_prompts: bool = field(
        default=get_env_value("ENABLE_DOMAIN_PROMPTS", False, bool)
    )
    """Enable domain-specific prompt templates for better accuracy."""

    enable_conversation_memory: bool = field(
        default=get_env_value("ENABLE_CONVERSATION_MEMORY", False, bool)
    )
    """Enable conversation history tracking for context-aware responses."""

    max_conversation_history: int = field(
        default=get_env_value("MAX_CONVERSATION_HISTORY", 5, int)
    )
    """Maximum number of conversation turns to retain in memory."""

    # Logging Configuration
    # ---
    log_query_improvements: bool = field(
        default=get_env_value("LOG_QUERY_IMPROVEMENTS", False, bool)
    )
    """Log query improvement operations for debugging and analysis."""

    log_verification_results: bool = field(
        default=get_env_value("LOG_VERIFICATION_RESULTS", False, bool)
    )
    """Log verification results and modification operations."""

    def __post_init__(self):
        """Post-initialization setup for backward compatibility and validation"""
        # Support legacy environment variable names for backward compatibility
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parse_method = legacy_parse_method
            import warnings

            warnings.warn(
                "MINERU_PARSE_METHOD is deprecated. Use PARSE_METHOD instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # NEW: Validate domain setting
        valid_domains = ["general", "medical", "legal", "financial", "technical", "academic"]
        if self.domain not in valid_domains:
            import warnings
            warnings.warn(
                f"Invalid domain '{self.domain}'. Must be one of {valid_domains}. "
                f"Using 'general' as default.",
                UserWarning,
                stacklevel=2,
            )
            self.domain = "general"

        # NEW: Validate query_improvement_method
        valid_methods = ["rewrite", "expand", "decompose", "llm", "rules", "hybrid"]
        if self.query_improvement_method not in valid_methods:
            import warnings
            warnings.warn(
                f"Invalid query_improvement_method '{self.query_improvement_method}'. "
                f"Must be one of {valid_methods}. Using 'rewrite' as default.",
                UserWarning,
                stacklevel=2,
            )
            self.query_improvement_method = "rewrite"

    @property
    def mineru_parse_method(self) -> str:
        """
        Backward compatibility property for old code.

        .. deprecated::
           Use `parse_method` instead. This property will be removed in a future version.
        """
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        """Setter for backward compatibility"""
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parse_method = value
