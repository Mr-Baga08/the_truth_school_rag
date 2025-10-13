"""
Streaming Query Module with Verification Support

This module provides streaming capabilities for RAGAnything while maintaining
the dual-LLM verification layer. It allows real-time token streaming to the
frontend while buffering the complete response for post-generation verification.

Key Features:
- Real-time token streaming from LLM (Gemini, OpenAI, etc.)
- Complete response buffering for verification
- Async verification after streaming completes
- Verification metadata injection into stream
- Support for both verified and unverified streaming modes

Architecture:
1. Stream tokens to frontend in real-time
2. Buffer complete response for verification
3. Run verification asynchronously after completion
4. Send verification metadata as final stream chunk

Author: RAG-Anything Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
from lightrag.utils import logger


# =============================================================================
# Configuration Classes
# =============================================================================

class StreamMode(Enum):
    """Streaming modes"""
    TOKENS_ONLY = "tokens_only"  # Stream tokens, no verification
    TOKENS_WITH_VERIFICATION = "tokens_with_verification"  # Stream tokens + verify
    TOKENS_WITH_METADATA = "tokens_with_metadata"  # Include metadata chunks


@dataclass
class StreamingConfig:
    """Configuration for streaming queries

    Attributes:
        mode: Streaming mode
        enable_verification: Whether to run verification after streaming
        send_verification_metadata: Send verification results as final chunk
        verification_async: Run verification in background (non-blocking)
        buffer_size: Number of tokens to buffer before sending
        include_context: Include retrieved context in metadata
    """
    mode: StreamMode = StreamMode.TOKENS_WITH_VERIFICATION
    enable_verification: bool = True
    send_verification_metadata: bool = True
    verification_async: bool = True
    buffer_size: int = 1
    include_context: bool = False


# =============================================================================
# Streaming Response Buffer
# =============================================================================

class StreamBuffer:
    """Buffer for collecting streamed tokens and managing verification

    This class collects tokens as they're streamed and provides the complete
    response for verification after streaming completes.
    """

    def __init__(self):
        """Initialize StreamBuffer"""
        self.tokens: List[str] = []
        self.complete_response: str = ""
        self.is_complete: bool = False
        self.verification_result: Optional[Dict[str, Any]] = None

    def add_token(self, token: str):
        """Add a token to the buffer

        Args:
            token: Token to add
        """
        self.tokens.append(token)

    def finalize(self) -> str:
        """Finalize buffer and return complete response

        Returns:
            Complete response string
        """
        self.complete_response = "".join(self.tokens)
        self.is_complete = True
        return self.complete_response

    def set_verification_result(self, result: Dict[str, Any]):
        """Store verification result

        Args:
            result: Verification result dictionary
        """
        self.verification_result = result


# =============================================================================
# Streaming Query Handler
# =============================================================================

class StreamingQueryHandler:
    """Handler for streaming queries with verification support

    This class orchestrates the streaming process, managing token streaming
    to the frontend while buffering for verification.

    Attributes:
        config: StreamingConfig instance
        verifier: AnswerVerifier instance (optional)
        modifier: AnswerModifier instance (optional)
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        verifier: Optional[Any] = None,
        modifier: Optional[Any] = None
    ):
        """Initialize StreamingQueryHandler

        Args:
            config: Streaming configuration
            verifier: AnswerVerifier instance for verification
            modifier: AnswerModifier instance for improvements
        """
        self.config = config or StreamingConfig()
        self.verifier = verifier
        self.modifier = modifier

    async def stream_with_verification(
        self,
        llm_stream_func: Callable,
        query: str,
        context: str,
        original_query: Optional[str] = None,
        **llm_kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream LLM response with verification support

        This is the main streaming method. It:
        1. Streams tokens to frontend in real-time
        2. Buffers tokens for complete response
        3. Runs verification after streaming completes
        4. Sends verification metadata as final chunk

        Args:
            llm_stream_func: Async generator function that yields tokens
            query: Query to answer
            context: Retrieved context
            original_query: Original query before improvement
            **llm_kwargs: Additional kwargs for LLM

        Yields:
            Dict with keys:
                - type: "token" | "metadata" | "verification" | "error"
                - content: Token string or metadata dict
                - done: Boolean indicating if streaming is complete

        Example:
            ```python
            async for chunk in handler.stream_with_verification(
                llm_stream_func=my_gemini_stream,
                query="What is photosynthesis?",
                context="[Retrieved context]"
            ):
                if chunk["type"] == "token":
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "verification":
                    print(f"\n\nVerification Score: {chunk['content']['score']}")
            ```
        """
        buffer = StreamBuffer()

        try:
            # Step 1: Stream tokens to frontend
            logger.info("Starting token streaming...")

            async for token in llm_stream_func(
                prompt=self._build_prompt(query, context),
                **llm_kwargs
            ):
                # Add token to buffer
                buffer.add_token(token)

                # Yield token to frontend
                yield {
                    "type": "token",
                    "content": token,
                    "done": False
                }

            # Step 2: Finalize buffer
            complete_response = buffer.finalize()
            logger.info(f"Streaming complete. Total response length: {len(complete_response)}")

            # Send completion signal
            yield {
                "type": "token",
                "content": "",
                "done": True
            }

            # Step 3: Run verification (if enabled)
            if self.config.enable_verification and self.verifier:
                logger.info("Running post-stream verification...")

                if self.config.verification_async:
                    # Non-blocking verification
                    asyncio.create_task(
                        self._verify_response_async(
                            buffer,
                            query,
                            context,
                            original_query
                        )
                    )

                    # Send placeholder verification metadata
                    if self.config.send_verification_metadata:
                        yield {
                            "type": "verification",
                            "content": {
                                "status": "verifying",
                                "message": "Verification in progress..."
                            },
                            "done": False
                        }
                else:
                    # Blocking verification
                    verification_result = await self._verify_response(
                        complete_response,
                        query,
                        context,
                        original_query
                    )
                    buffer.set_verification_result(verification_result)

                    # Send verification metadata
                    if self.config.send_verification_metadata:
                        yield {
                            "type": "verification",
                            "content": verification_result,
                            "done": True
                        }

        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": {
                    "message": str(e),
                    "error_type": type(e).__name__
                },
                "done": True
            }

    async def stream_simple(
        self,
        llm_stream_func: Callable,
        query: str,
        context: str,
        **llm_kwargs
    ) -> AsyncGenerator[str, None]:
        """Simple token streaming without verification

        This is a lightweight streaming method that just yields tokens
        without any verification or metadata.

        Args:
            llm_stream_func: Async generator function that yields tokens
            query: Query to answer
            context: Retrieved context
            **llm_kwargs: Additional kwargs for LLM

        Yields:
            str: Individual tokens

        Example:
            ```python
            async for token in handler.stream_simple(
                llm_stream_func=my_llm_stream,
                query="What is AI?",
                context="[Context]"
            ):
                print(token, end="", flush=True)
            ```
        """
        try:
            async for token in llm_stream_func(
                prompt=self._build_prompt(query, context),
                **llm_kwargs
            ):
                yield token

        except Exception as e:
            logger.error(f"Error during simple streaming: {e}", exc_info=True)
            yield f"[Error: {str(e)}]"

    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt from query and context

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        # Enhanced prompt with better instructions for higher quality responses
        return f"""You are an expert assistant analyzing a knowledge base. Use the provided context to answer the question accurately and comprehensively.

## Context Information:
{context}

## User Question:
{query}

## Instructions:
1. Answer based ONLY on the information provided in the context above
2. If the context contains relevant information, provide a clear, detailed answer
3. Structure your response with:
   - Direct answer to the question
   - Supporting details and evidence from the context
   - Relevant examples or specifics when available
4. If the context doesn't contain enough information to fully answer the question, state what you know and what's missing
5. Be precise and cite specific information from the context when possible
6. Use clear, professional language appropriate for the domain

## Answer:"""

    async def _verify_response(
        self,
        response: str,
        query: str,
        context: str,
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify a complete response

        Args:
            response: Complete LLM response
            query: Query used
            context: Retrieved context
            original_query: Original query before improvement

        Returns:
            Verification result dictionary
        """
        if not self.verifier:
            logger.warning("Verifier not available, skipping verification")
            return {
                "passed": True,
                "score": 10.0,
                "message": "Verification not available"
            }

        try:
            verification_result = await self.verifier.verify_answer(
                query=query,
                answer=response,
                context=context,
                original_query=original_query
            )

            return {
                "passed": verification_result.get("passed", False),
                "score": verification_result.get("overall_score", 0.0),
                "criteria_scores": verification_result.get("criteria_scores", {}),
                "issues": verification_result.get("issues", []),
                "suggestions": verification_result.get("suggestions", []),
                "confidence": verification_result.get("confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"Verification error: {e}", exc_info=True)
            return {
                "passed": False,
                "score": 0.0,
                "error": str(e)
            }

    async def _verify_response_async(
        self,
        buffer: StreamBuffer,
        query: str,
        context: str,
        original_query: Optional[str] = None
    ):
        """Async verification (non-blocking background task)

        Args:
            buffer: StreamBuffer to store result in
            query: Query used
            context: Retrieved context
            original_query: Original query before improvement
        """
        verification_result = await self._verify_response(
            buffer.complete_response,
            query,
            context,
            original_query
        )
        buffer.set_verification_result(verification_result)
        logger.info(f"Background verification complete: score={verification_result.get('score', 0):.2f}")


# =============================================================================
# Streaming Mixin for RAGAnything Integration
# =============================================================================

class StreamingQueryMixin:
    """Mixin providing streaming query capabilities to RAGAnything

    This mixin adds streaming query methods that can be used alongside
    the existing query methods. It integrates with the verification system.

    Expected attributes:
    - self.lightrag: LightRAG instance
    - self.answer_verifier: AnswerVerifier instance (optional)
    - self.answer_modifier: AnswerModifier instance (optional)
    - self.config: RAGAnythingConfig instance
    - self.logger: Logger instance
    """

    async def aquery_stream(
        self,
        query: str,
        mode: str = "mix",
        enable_verification: bool = True,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming query with verification support

        This method streams LLM responses while optionally running verification.
        Perfect for real-time user interfaces.

        Args:
            query: User query
            mode: RAG mode ("local", "global", "hybrid", "naive", "mix")
            enable_verification: Whether to run verification
            **kwargs: Additional query parameters

        Yields:
            Dict containing:
                - type: "token" | "metadata" | "verification" | "error"
                - content: Token or metadata
                - done: Completion flag

        Example:
            ```python
            async for chunk in rag.aquery_stream(
                query="What is machine learning?",
                enable_verification=True
            ):
                if chunk["type"] == "token":
                    print(chunk["content"], end="")
                elif chunk["type"] == "verification":
                    print(f"\n\nQuality Score: {chunk['content']['score']}/10")
            ```
        """
        if not hasattr(self, 'lightrag') or self.lightrag is None:
            raise ValueError("LightRAG not initialized")

        try:
            # Import here to avoid circular dependencies
            from lightrag import QueryParam

            original_query = query

            # Step 1: Apply query improvement if enabled
            use_query_improvement = kwargs.pop(
                'enable_query_improvement',
                getattr(self.config, 'enable_query_improvement', False)
            )

            if use_query_improvement and hasattr(self, 'query_improver') and self.query_improver:
                self.logger.info("Applying query improvement for streaming...")
                try:
                    query_improvement_result = await self._apply_query_improvement(query)
                    improved = query_improvement_result.get("improved_query", query)
                    if improved and improved.strip():
                        query = improved
                        self.logger.info(f"Query improved: '{original_query}' -> '{query}'")
                    else:
                        self.logger.warning("Query improvement returned empty result, using original query")
                except Exception as e:
                    self.logger.warning(f"Query improvement failed: {e}, using original query")
                    # Continue with original query on error

            # Step 2: Retrieve context
            self.logger.info(f"Retrieving context for streaming query: {query[:100]}...")
            query_param = QueryParam(mode=mode, only_need_context=True)
            context = await self.lightrag.aquery(query, param=query_param)

            if not context or not context.strip():
                self.logger.warning("No context retrieved for query")
                yield {
                    "type": "error",
                    "content": {
                        "message": "I couldn't find any relevant information in the knowledge base to answer your question. Please ensure documents have been uploaded and indexed, or try rephrasing your query with different keywords.",
                        "suggestion": "Try uploading relevant documents first, or rephrase your question with more specific terms."
                    },
                    "done": True
                }
                return

            # Step 3: Create streaming handler
            streaming_config = StreamingConfig(
                enable_verification=enable_verification and hasattr(self, 'answer_verifier'),
                send_verification_metadata=True,
                verification_async=False  # Blocking to ensure verification completes
            )

            handler = StreamingQueryHandler(
                config=streaming_config,
                verifier=getattr(self, 'answer_verifier', None),
                modifier=getattr(self, 'answer_modifier', None)
            )

            # Step 4: Stream response
            if hasattr(self.lightrag, 'llm_model_func'):
                # Create streaming wrapper for non-streaming LLM
                llm_func = self.lightrag.llm_model_func

                async def llm_stream_wrapper(prompt, **llm_kwargs):
                    """Wrapper to simulate streaming from non-streaming LLM"""
                    if asyncio.iscoroutinefunction(llm_func):
                        response = await llm_func(prompt, **llm_kwargs)
                    else:
                        response = llm_func(prompt, **llm_kwargs)

                    # Simulate token-by-token streaming
                    # Split by words for more natural streaming
                    words = response.split()
                    for i, word in enumerate(words):
                        if i < len(words) - 1:
                            yield word + " "
                        else:
                            yield word
                        # Small delay to simulate real streaming
                        await asyncio.sleep(0.01)

                async for chunk in handler.stream_with_verification(
                    llm_stream_func=llm_stream_wrapper,
                    query=query,
                    context=context,
                    original_query=original_query
                ):
                    yield chunk
            else:
                raise ValueError("LLM model function not available for streaming")

        except Exception as e:
            self.logger.error(f"Error in streaming query: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": {"message": str(e)},
                "done": True
            }

    async def aquery_stream_simple(
        self,
        query: str,
        mode: str = "mix",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Simple streaming query without verification

        Lightweight streaming that just yields tokens without any
        verification or metadata overhead.

        Args:
            query: User query
            mode: RAG mode
            **kwargs: Additional parameters

        Yields:
            str: Individual tokens

        Example:
            ```python
            async for token in rag.aquery_stream_simple(
                query="Explain photosynthesis"
            ):
                print(token, end="", flush=True)
            ```
        """
        try:
            # Get context
            from lightrag import QueryParam

            query_param = QueryParam(mode=mode, only_need_context=True)
            context = await self.lightrag.aquery(query, param=query_param)

            if not context:
                yield "[No context found]"
                return

            # Create handler
            handler = StreamingQueryHandler(
                config=StreamingConfig(enable_verification=False)
            )

            # Stream tokens
            if hasattr(self.lightrag, 'llm_model_func'):
                llm_func = self.lightrag.llm_model_func

                async def llm_stream_wrapper(prompt, **llm_kwargs):
                    if asyncio.iscoroutinefunction(llm_func):
                        response = await llm_func(prompt, **llm_kwargs)
                    else:
                        response = llm_func(prompt, **llm_kwargs)

                    words = response.split()
                    for i, word in enumerate(words):
                        if i < len(words) - 1:
                            yield word + " "
                        else:
                            yield word
                        await asyncio.sleep(0.01)

                async for token in handler.stream_simple(
                    llm_stream_func=llm_stream_wrapper,
                    query=query,
                    context=context
                ):
                    yield token
            else:
                yield "[LLM not available]"

        except Exception as e:
            self.logger.error(f"Error in simple streaming: {e}", exc_info=True)
            yield f"[Error: {str(e)}]"
