"""
Complete document parsing + multimodal content insertion Pipeline

This script integrates:
1. Document parsing (using configurable parsers)
2. Pure text content LightRAG insertion
3. Specialized processing for multimodal content (using different processors)
"""

import os
from typing import Dict, Any, Optional, Callable
import sys
import asyncio
import atexit
from dataclasses import dataclass, field
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag import LightRAG
from lightrag.utils import logger
from dotenv import load_dotenv

# Load environment variables from .env file
# The OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Import configuration and modules
from raganything.config import RAGAnythingConfig
from raganything.query import QueryMixin
from raganything.processor import ProcessorMixin
from raganything.batch import BatchMixin
from raganything.utils import get_processor_supports
from raganything.parser import MineruParser, DoclingParser

# Import specialized processors
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
    ContextExtractor,
    ContextConfig,
)

# Import enhancement modules
from raganything.query_improvement import (
    QueryImprover,
    QueryRewriter,
    QueryImprovementConfig,
)
from raganything.verification import (
    DualLLMPipeline,
    VerificationConfig,
    AnswerVerifier,
    AnswerModifier,
)


@dataclass
class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin):
    """Multimodal Document Processing Pipeline - Complete document parsing and insertion pipeline"""

    # Core Components
    # ---
    lightrag: Optional[LightRAG] = field(default=None)
    """Optional pre-initialized LightRAG instance."""

    llm_model_func: Optional[Callable] = field(default=None)
    """LLM model function for text analysis."""

    verifier_llm_func: Optional[Callable] = field(default=None)
    """Optional separate LLM function for answer verification (typically more powerful than generator)."""

    vision_model_func: Optional[Callable] = field(default=None)
    """Vision model function for image analysis."""

    embedding_func: Optional[Callable] = field(default=None)
    """Embedding function for text vectorization."""

    config: Optional[RAGAnythingConfig] = field(default=None)
    """Configuration object, if None will create with environment variables."""

    # LightRAG Configuration
    # ---
    lightrag_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for LightRAG initialization when lightrag is not provided.
    This allows passing all LightRAG configuration parameters like:
    - kv_storage, vector_storage, graph_storage, doc_status_storage
    - top_k, chunk_top_k, max_entity_tokens, max_relation_tokens, max_total_tokens
    - cosine_threshold, related_chunk_number
    - chunk_token_size, chunk_overlap_token_size, tokenizer, tiktoken_model_name
    - embedding_batch_num, embedding_func_max_async, embedding_cache_config
    - llm_model_name, llm_model_max_token_size, llm_model_max_async, llm_model_kwargs
    - rerank_model_func, vector_db_storage_cls_kwargs, enable_llm_cache
    - max_parallel_insert, max_graph_nodes, addon_params, etc.
    """

    # Internal State
    # ---
    modal_processors: Dict[str, Any] = field(default_factory=dict, init=False)
    """Dictionary of multimodal processors."""

    context_extractor: Optional[ContextExtractor] = field(default=None, init=False)
    """Context extractor for providing surrounding content to modal processors."""

    parse_cache: Optional[Any] = field(default=None, init=False)
    """Parse result cache storage using LightRAG KV storage."""

    _parser_installation_checked: bool = field(default=False, init=False)
    """Flag to track if parser installation has been checked."""

    # Enhancement Components (Query Improvement & Verification)
    # ---
    query_improver: Optional[QueryImprover] = field(default=None, init=False)
    """Query improvement component for enhancing user queries."""

    query_rewriter: Optional[QueryRewriter] = field(default=None, init=False)
    """Query rewriting component for conversation context handling."""

    answer_verifier: Optional[AnswerVerifier] = field(default=None, init=False)
    """Answer verification component for quality assessment."""

    answer_modifier: Optional[AnswerModifier] = field(default=None, init=False)
    """Answer modification component for iterative improvement."""

    verification_pipeline: Optional[DualLLMPipeline] = field(default=None, init=False)
    """Dual-LLM verification pipeline orchestrating verification and modification."""

    conversation_history: Dict[str, list] = field(default_factory=dict, init=False)
    """Conversation history storage for context-aware query processing."""

    _enhancements_initialized: bool = field(default=False, init=False)
    """Flag to track if enhancement components have been initialized."""

    def __post_init__(self):
        """Post-initialization setup following LightRAG pattern"""
        # Initialize configuration if not provided
        if self.config is None:
            self.config = RAGAnythingConfig()

        # Set working directory
        self.working_dir = self.config.working_dir

        # Set up logger (use existing logger, don't configure it)
        self.logger = logger

        # Set up document parser
        self.doc_parser = (
            DoclingParser() if self.config.parser == "docling" else MineruParser()
        )

        # Register close method for cleanup
        atexit.register(self.close)

        # Create working directory if needed
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        # Log configuration info
        self.logger.info("RAGAnything initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(f"  Parser: {self.config.parser}")
        self.logger.info(f"  Parse method: {self.config.parse_method}")
        self.logger.info(
            f"  Multimodal processing - Image: {self.config.enable_image_processing}, "
            f"Table: {self.config.enable_table_processing}, "
            f"Equation: {self.config.enable_equation_processing}"
        )
        self.logger.info(f"  Max concurrent files: {self.config.max_concurrent_files}")

        # Log enhancement features status
        if self.config.enable_query_improvement:
            self.logger.info(
                f"  Query improvement: ENABLED (method: {self.config.query_improvement_method})"
            )
        if self.config.enable_dual_llm_verification or getattr(self.config, 'enable_answer_verification', False):
            self.logger.info(
                f"  Answer verification: ENABLED (threshold: {self.config.verification_threshold})"
            )
        if getattr(self.config, 'enable_conversation_memory', False):
            self.logger.info(
                f"  Conversation memory: ENABLED (max history: {getattr(self.config, 'max_conversation_history', 5)})"
            )

    def close(self):
        """Cleanup resources when object is destroyed"""
        try:
            import asyncio

            if asyncio.get_event_loop().is_running():
                # If we're in an async context, schedule cleanup
                asyncio.create_task(self.finalize_storages())
            else:
                # Run cleanup synchronously
                asyncio.run(self.finalize_storages())
        except Exception as e:
            # Use print instead of logger since logger might be cleaned up already
            print(f"Warning: Failed to finalize RAGAnything storages: {e}")

    def _create_context_config(self) -> ContextConfig:
        """Create context configuration from RAGAnything config"""
        return ContextConfig(
            context_window=self.config.context_window,
            context_mode=self.config.context_mode,
            max_context_tokens=self.config.max_context_tokens,
            include_headers=self.config.include_headers,
            include_captions=self.config.include_captions,
            filter_content_types=self.config.context_filter_content_types,
        )

    def _create_context_extractor(self) -> ContextExtractor:
        """Create context extractor with tokenizer from LightRAG"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG must be initialized before creating context extractor"
            )

        context_config = self._create_context_config()
        return ContextExtractor(
            config=context_config, tokenizer=self.lightrag.tokenizer
        )

    def _initialize_processors(self):
        """Initialize multimodal processors with appropriate model functions"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG instance must be initialized before creating processors"
            )

        # Create context extractor
        self.context_extractor = self._create_context_extractor()

        # Create different multimodal processors based on configuration
        self.modal_processors = {}

        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_equation_processing:
            self.modal_processors["equation"] = EquationModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        # Always include generic processor as fallback
        self.modal_processors["generic"] = GenericModalProcessor(
            lightrag=self.lightrag,
            modal_caption_func=self.llm_model_func,
            context_extractor=self.context_extractor,
        )

        self.logger.info("Multimodal processors initialized with context support")
        self.logger.info(f"Available processors: {list(self.modal_processors.keys())}")
        self.logger.info(f"Context configuration: {self._create_context_config()}")

    async def initialize_enhancements(self):
        """
        Initialize query improvement and verification components

        This method initializes the enhancement layer including:
        - Query improvement (rewriting, expansion, entity extraction)
        - Dual-LLM verification (answer quality assessment)
        - Conversation memory management

        Should be called after LightRAG initialization.
        """
        if self._enhancements_initialized:
            self.logger.debug("Enhancement components already initialized")
            return

        try:
            # Initialize query improvement components
            if self.config.enable_query_improvement and self.llm_model_func:
                self.logger.info("Initializing query improvement components...")

                # Create query improvement configuration
                query_improvement_config = QueryImprovementConfig(
                    method=self.config.query_improvement_method,
                    domain=getattr(self.config, 'domain', 'general'),
                    expand_abbreviations=getattr(self.config, 'expand_abbreviations', False),
                    add_domain_keywords=getattr(self.config, 'add_domain_keywords', False),
                    extract_entities=getattr(self.config, 'extract_query_entities', False),
                    llm_temperature=0.3,
                    llm_max_tokens=500,
                )

                # Initialize QueryImprover
                self.query_improver = QueryImprover(
                    llm_func=self.llm_model_func,
                    config=query_improvement_config,
                )

                # Initialize QueryRewriter if conversation memory is enabled
                if getattr(self.config, 'enable_query_rewriting', False) or \
                   getattr(self.config, 'enable_conversation_memory', False):
                    self.query_rewriter = QueryRewriter(
                        llm_func=self.llm_model_func,
                        config=query_improvement_config,
                    )
                    self.logger.info("Query rewriter initialized with conversation support")

                self.logger.info(
                    f"Query improvement initialized (method: {self.config.query_improvement_method}, "
                    f"domain: {getattr(self.config, 'domain', 'general')})"
                )

            # Initialize verification components
            enable_verification = (
                self.config.enable_dual_llm_verification or
                getattr(self.config, 'enable_answer_verification', False)
            )

            if enable_verification and self.llm_model_func:
                self.logger.info("Initializing dual-LLM verification components...")

                # Determine which LLM to use for verification
                verifier_llm = self.verifier_llm_func or self.llm_model_func

                # Create verification configuration
                verification_config = VerificationConfig(
                    verification_threshold=self.config.verification_threshold,
                    max_modification_iterations=getattr(
                        self.config, 'max_verification_iterations',
                        self.config.max_verification_retries
                    ),
                    require_all_criteria_pass=False,
                    individual_criterion_threshold=6.0,
                    enable_confidence_scoring=getattr(self.config, 'add_confidence_score', True),
                    enable_detailed_feedback=True,
                    stop_on_first_pass=True,
                    min_improvement_delta=0.5,
                )

                # Initialize AnswerVerifier
                self.answer_verifier = AnswerVerifier(
                    verifier_llm_func=verifier_llm,
                    config=verification_config,
                )

                # Initialize AnswerModifier if enabled
                if getattr(self.config, 'enable_answer_modification', False):
                    self.answer_modifier = AnswerModifier(
                        generator_llm_func=self.llm_model_func,
                        config=verification_config,
                    )

                    # Initialize DualLLMPipeline for orchestration
                    self.verification_pipeline = DualLLMPipeline(
                        generator_llm=self.llm_model_func,
                        verifier_llm=verifier_llm,
                        config=verification_config,
                    )
                    self.logger.info("Verification pipeline initialized with answer modification")
                else:
                    self.logger.info("Answer verifier initialized (modification disabled)")

                self.logger.info(
                    f"Verification initialized (threshold: {self.config.verification_threshold}, "
                    f"max iterations: {getattr(self.config, 'max_verification_iterations', self.config.max_verification_retries)})"
                )

            self._enhancements_initialized = True
            self.logger.info("All enhancement components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize enhancement components: {e}", exc_info=True)
            # Don't raise - allow RAGAnything to work without enhancements
            self.logger.warning("Continuing without enhancement features")

    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    async def _ensure_lightrag_initialized(self):
        """Ensure LightRAG instance is initialized, create if necessary"""
        try:
            # Check parser installation first
            if not self._parser_installation_checked:
                if not self.doc_parser.check_installation():
                    error_msg = (
                        f"Parser '{self.config.parser}' is not properly installed. "
                        "Please install it using 'pip install' or 'uv pip install'."
                    )
                    self.logger.error(error_msg)
                    return {"success": False, "error": error_msg}

                self._parser_installation_checked = True
                self.logger.info(f"Parser '{self.config.parser}' installation verified")

            if self.lightrag is not None:
                # LightRAG was pre-provided, but we need to ensure it's properly initialized
                try:
                    # Ensure LightRAG storages are initialized
                    if (
                        not hasattr(self.lightrag, "_storages_status")
                        or self.lightrag._storages_status.name != "INITIALIZED"
                    ):
                        self.logger.info(
                            "Initializing storages for pre-provided LightRAG instance"
                        )
                        await self.lightrag.initialize_storages()
                        from lightrag.kg.shared_storage import (
                            initialize_pipeline_status,
                        )

                        await initialize_pipeline_status()

                    # Initialize parse cache if not already done
                    if self.parse_cache is None:
                        self.logger.info(
                            "Initializing parse cache for pre-provided LightRAG instance"
                        )
                        self.parse_cache = (
                            self.lightrag.key_string_value_json_storage_cls(
                                namespace="parse_cache",
                                workspace=self.lightrag.workspace,
                                global_config=self.lightrag.__dict__,
                                embedding_func=self.embedding_func,
                            )
                        )
                        await self.parse_cache.initialize()

                    # Initialize processors if not already done
                    if not self.modal_processors:
                        self._initialize_processors()

                    # Initialize enhancement components
                    await self.initialize_enhancements()

                    return {"success": True}

                except Exception as e:
                    error_msg = (
                        f"Failed to initialize pre-provided LightRAG instance: {str(e)}"
                    )
                    self.logger.error(error_msg, exc_info=True)
                    return {"success": False, "error": error_msg}

            # Validate required functions for creating new LightRAG instance
            if self.llm_model_func is None:
                error_msg = "llm_model_func must be provided when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                error_msg = "embedding_func must be provided when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            from lightrag.kg.shared_storage import initialize_pipeline_status

            # Prepare LightRAG initialization parameters
            lightrag_params = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
            }

            # Merge user-provided lightrag_kwargs, which can override defaults
            lightrag_params.update(self.lightrag_kwargs)

            # Log the parameters being used for initialization (excluding sensitive data)
            log_params = {
                k: v
                for k, v in lightrag_params.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            self.logger.info(f"Initializing LightRAG with parameters: {log_params}")

            try:
                # Create LightRAG instance with merged parameters
                self.lightrag = LightRAG(**lightrag_params)
                await self.lightrag.initialize_storages()
                await initialize_pipeline_status()

                # Initialize parse cache storage using LightRAG's KV storage
                self.parse_cache = self.lightrag.key_string_value_json_storage_cls(
                    namespace="parse_cache",
                    workspace=self.lightrag.workspace,
                    global_config=self.lightrag.__dict__,
                    embedding_func=self.embedding_func,
                )
                await self.parse_cache.initialize()

                # Initialize processors after LightRAG is ready
                self._initialize_processors()

                # Initialize enhancement components
                await self.initialize_enhancements()

                self.logger.info(
                    "LightRAG, parse cache, and multimodal processors initialized"
                )
                return {"success": True}

            except Exception as e:
                error_msg = f"Failed to initialize LightRAG instance: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error during LightRAG initialization: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}

    async def finalize_storages(self):
        """Finalize all storages including parse cache and LightRAG storages

        This method should be called when shutting down to properly clean up resources
        and persist any cached data. It will finalize both the parse cache and LightRAG's
        internal storages.

        Example usage:
            try:
                rag_anything = RAGAnything(...)
                await rag_anything.process_file("document.pdf")
                # ... other operations ...
            finally:
                # Always finalize storages to clean up resources
                if rag_anything:
                    await rag_anything.finalize_storages()

        Note:
            - This method is automatically called in __del__ when the object is destroyed
            - Manual calling is recommended in production environments
            - All finalization tasks run concurrently for better performance
        """
        try:
            tasks = []

            # Finalize parse cache if it exists
            if self.parse_cache is not None:
                tasks.append(self.parse_cache.finalize())
                self.logger.debug("Scheduled parse cache finalization")

            # Finalize LightRAG storages if LightRAG is initialized
            if self.lightrag is not None:
                tasks.append(self.lightrag.finalize_storages())
                self.logger.debug("Scheduled LightRAG storages finalization")

            # Run all finalization tasks concurrently
            if tasks:
                await asyncio.gather(*tasks)
                self.logger.info("Successfully finalized all RAGAnything storages")
            else:
                self.logger.debug("No storages to finalize")

        except Exception as e:
            self.logger.error(f"Error during storage finalization: {e}")
            raise

    def check_parser_installation(self) -> bool:
        """
        Check if the configured parser is properly installed

        Returns:
            bool: True if the configured parser is properly installed
        """
        return self.doc_parser.check_installation()

    def verify_parser_installation_once(self) -> bool:
        if not self._parser_installation_checked:
            if not self.doc_parser.check_installation():
                raise RuntimeError(
                    f"Parser '{self.config.parser}' is not properly installed. "
                    "Please install it using pip install or uv pip install."
                )
            self._parser_installation_checked = True
            self.logger.info(f"Parser '{self.config.parser}' installation verified")
        return True

    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        config_info = {
            "directory": {
                "working_dir": self.config.working_dir,
                "parser_output_dir": self.config.parser_output_dir,
            },
            "parsing": {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
                "display_content_stats": self.config.display_content_stats,
            },
            "multimodal_processing": {
                "enable_image_processing": self.config.enable_image_processing,
                "enable_table_processing": self.config.enable_table_processing,
                "enable_equation_processing": self.config.enable_equation_processing,
            },
            "context_extraction": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
                "max_context_tokens": self.config.max_context_tokens,
                "include_headers": self.config.include_headers,
                "include_captions": self.config.include_captions,
                "filter_content_types": self.config.context_filter_content_types,
            },
            "batch_processing": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "supported_file_extensions": self.config.supported_file_extensions,
                "recursive_folder_processing": self.config.recursive_folder_processing,
            },
            "logging": {
                "note": "Logging fields have been removed - configure logging externally",
            },
        }

        # Add LightRAG configuration if available
        if self.lightrag_kwargs:
            # Filter out sensitive data and callable objects for display
            safe_kwargs = {
                k: v
                for k, v in self.lightrag_kwargs.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            config_info["lightrag_config"] = {
                "custom_parameters": safe_kwargs,
                "note": "LightRAG will be initialized with these additional parameters",
            }
        else:
            config_info["lightrag_config"] = {
                "custom_parameters": {},
                "note": "Using default LightRAG parameters",
            }

        return config_info

    def set_content_source_for_context(
        self, content_source, content_format: str = "auto"
    ):
        """Set content source for context extraction in all modal processors

        Args:
            content_source: Source content for context extraction (e.g., MinerU content list)
            content_format: Format of content source ("minerU", "text_chunks", "auto")
        """
        if not self.modal_processors:
            self.logger.warning(
                "Modal processors not initialized. Content source will be set when processors are created."
            )
            return

        for processor_name, processor in self.modal_processors.items():
            try:
                processor.set_content_source(content_source, content_format)
                self.logger.debug(f"Set content source for {processor_name} processor")
            except Exception as e:
                self.logger.error(
                    f"Failed to set content source for {processor_name}: {e}"
                )

        self.logger.info(
            f"Content source set for context extraction (format: {content_format})"
        )

    def update_context_config(self, **context_kwargs):
        """Update context extraction configuration

        Args:
            **context_kwargs: Context configuration parameters to update
                (context_window, context_mode, max_context_tokens, etc.)
        """
        # Update the main config
        for key, value in context_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated context config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown context config parameter: {key}")

        # Recreate context extractor with new config if processors are initialized
        if self.lightrag and self.modal_processors:
            try:
                self.context_extractor = self._create_context_extractor()
                # Update all processors with new context extractor
                for processor_name, processor in self.modal_processors.items():
                    processor.context_extractor = self.context_extractor

                self.logger.info(
                    "Context configuration updated and applied to all processors"
                )
                self.logger.info(
                    f"New context configuration: {self._create_context_config()}"
                )
            except Exception as e:
                self.logger.error(f"Failed to update context configuration: {e}")

    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "config": self.get_config_info(),
            "models": {
                "llm_model": "External function"
                if self.llm_model_func
                else "Not provided",
                "vision_model": "External function"
                if self.vision_model_func
                else "Not provided",
                "embedding_model": "External function"
                if self.embedding_func
                else "Not provided",
            },
        }

        if not self.modal_processors:
            base_info["status"] = "Not initialized"
            base_info["processors"] = {}
        else:
            base_info["status"] = "Initialized"
            base_info["processors"] = {}

            for proc_type, processor in self.modal_processors.items():
                base_info["processors"][proc_type] = {
                    "class": processor.__class__.__name__,
                    "supports": get_processor_supports(proc_type),
                    "enabled": True,
                }

        return base_info


async def create_rag_anything(
    llm_model_func: Callable,
    vision_model_func: Optional[Callable] = None,
    embedding_func: Optional[Callable] = None,
    verifier_llm_func: Optional[Callable] = None,
    config: Optional[RAGAnythingConfig] = None,
    **lightrag_kwargs
) -> RAGAnything:
    """
    Factory function to create and initialize a RAGAnything instance

    This is the recommended way to create a RAGAnything instance as it ensures
    all components are properly initialized, including query improvement and
    verification enhancements.

    Args:
        llm_model_func: LLM model function for text generation and analysis
        vision_model_func: Optional vision model function for image analysis
        embedding_func: Embedding function for text vectorization
        verifier_llm_func: Optional separate LLM for verification (typically more powerful)
        config: Optional configuration object. If None, uses environment variables
        **lightrag_kwargs: Additional parameters passed to LightRAG initialization

    Returns:
        RAGAnything: Fully initialized RAGAnything instance

    Example:
        ```python
        from raganything import create_rag_anything, RAGAnythingConfig

        # Create with custom config
        config = RAGAnythingConfig(
            enable_query_improvement=True,
            query_improvement_method="hybrid",
            enable_dual_llm_verification=True,
            verification_threshold=7.5,
            domain="medical"
        )

        rag = await create_rag_anything(
            llm_model_func=my_llm_func,
            vision_model_func=my_vision_func,
            embedding_func=my_embedding_func,
            verifier_llm_func=my_verifier_func,  # Optional stronger model
            config=config
        )

        # Process documents
        await rag.process_file("medical_paper.pdf")

        # Query with enhancements
        result = await rag.aquery(
            "What are the treatment options for hypertension?",
            enable_query_improvement=True,
            enable_verification=True,
            return_verification_info=True
        )
        ```
    """
    # Create RAGAnything instance
    rag = RAGAnything(
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        verifier_llm_func=verifier_llm_func,
        config=config,
        lightrag_kwargs=lightrag_kwargs,
    )

    # Initialize LightRAG and all components (including enhancements)
    init_result = await rag._ensure_lightrag_initialized()

    if not init_result.get("success", False):
        error_msg = init_result.get("error", "Unknown initialization error")
        raise RuntimeError(f"Failed to initialize RAGAnything: {error_msg}")

    rag.logger.info("RAGAnything created and fully initialized")

    return rag
