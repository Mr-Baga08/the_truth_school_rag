"""
Reranking Module for RAG-Anything

Provides reranking functionality using:
1. Gemini-based LLM reranking (free tier compatible)
2. Cross-encoder scoring
3. Relevance-based reordering

Author: RAG-Anything Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiReranker:
    """Reranker using Gemini API for semantic relevance scoring"""

    def __init__(
        self,
        llm_func: Optional[Callable] = None,
        model_name: str = "models/gemini-2.5-flash",
        batch_size: int = 5,
        temperature: float = 0.1
    ):
        """
        Initialize Gemini-based reranker

        Args:
            llm_func: Optional LLM function to use for reranking
            model_name: Gemini model to use
            batch_size: Number of chunks to process in parallel
            temperature: Temperature for relevance scoring
        """
        self.llm_func = llm_func
        self.model_name = model_name
        self.batch_size = batch_size
        self.temperature = temperature

    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on relevance to query

        Args:
            query: Search query
            chunks: List of chunks with 'content' field
            top_k: Return only top K results (None = return all, reranked)

        Returns:
            List of reranked chunks with 'relevance_score' field added
        """
        if not chunks:
            logger.warning("No chunks to rerank")
            return []

        if len(chunks) == 1:
            logger.debug("Only one chunk, skipping reranking")
            chunks[0]['relevance_score'] = 1.0
            return chunks

        logger.info(f"Reranking {len(chunks)} chunks for query: {query[:50]}...")

        try:
            # Score all chunks
            scored_chunks = await self._score_chunks_batch(query, chunks)

            # Sort by relevance score (descending)
            scored_chunks.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)

            # Return top_k if specified
            if top_k:
                scored_chunks = scored_chunks[:top_k]

            logger.info(f"Reranking complete. Top score: {scored_chunks[0].get('relevance_score', 0.0):.3f}")
            return scored_chunks

        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            # Return original chunks on error
            for chunk in chunks:
                chunk['relevance_score'] = 0.5  # Neutral score
            return chunks

    async def _score_chunks_batch(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score chunks in batches for efficiency"""

        # Process in batches to avoid rate limits
        results = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]

            # Score batch in parallel
            tasks = [self._score_single_chunk(query, chunk) for chunk in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Small delay between batches to avoid rate limiting
            if i + self.batch_size < len(chunks):
                await asyncio.sleep(0.2)

        return results

    async def _score_single_chunk(
        self,
        query: str,
        chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Score a single chunk for relevance"""

        content = chunk.get('content', '')
        if not content:
            chunk['relevance_score'] = 0.0
            return chunk

        try:
            # Build scoring prompt
            prompt = self._build_scoring_prompt(query, content)

            # Call LLM for scoring
            if self.llm_func:
                response = await self._call_llm_safely(prompt)
            else:
                response = await self._call_gemini_directly(prompt)

            # Parse score from response
            score = self._parse_score(response)
            chunk['relevance_score'] = score

            return chunk

        except Exception as e:
            logger.debug(f"Error scoring chunk: {e}")
            chunk['relevance_score'] = 0.5  # Neutral score on error
            return chunk

    def _build_scoring_prompt(self, query: str, content: str) -> str:
        """Build prompt for relevance scoring"""

        # Truncate content if too long
        max_content_len = 1000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        prompt = f"""Rate the relevance of the following text passage to the given query on a scale of 0.0 to 1.0.

Query: {query}

Passage:
{content}

Instructions:
- Consider semantic relevance, not just keyword matching
- 1.0 = Highly relevant, directly answers the query
- 0.7-0.9 = Relevant, contains useful information
- 0.4-0.6 = Somewhat relevant, tangentially related
- 0.0-0.3 = Not relevant or off-topic

Return ONLY a single number between 0.0 and 1.0, nothing else."""

        return prompt

    async def _call_llm_safely(self, prompt: str) -> str:
        """Call LLM function safely with error handling"""

        try:
            if asyncio.iscoroutinefunction(self.llm_func):
                response = await self.llm_func(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=10
                )
            else:
                response = self.llm_func(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=10
                )
            return response

        except Exception as e:
            logger.debug(f"Error calling LLM: {e}")
            raise

    async def _call_gemini_directly(self, prompt: str) -> str:
        """Call Gemini API directly as fallback"""

        try:
            model = genai.GenerativeModel(self.model_name)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=10
                )
            )
            return response.text

        except Exception as e:
            logger.debug(f"Error calling Gemini directly: {e}")
            raise

    def _parse_score(self, response: str) -> float:
        """Parse relevance score from LLM response"""

        try:
            # Extract first number from response
            import re
            numbers = re.findall(r'0?\.\d+|[01]\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                # Clamp to valid range
                return max(0.0, min(1.0, score))
            else:
                logger.debug(f"Could not parse score from: {response}")
                return 0.5

        except Exception as e:
            logger.debug(f"Error parsing score: {e}")
            return 0.5


def create_gemini_reranker(
    llm_func: Optional[Callable] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> GeminiReranker:
    """
    Factory function to create a Gemini reranker

    Args:
        llm_func: Optional LLM function to use
        api_key: Gemini API key (if calling directly)
        **kwargs: Additional arguments for GeminiReranker

    Returns:
        Configured GeminiReranker instance
    """
    if api_key:
        genai.configure(api_key=api_key)

    return GeminiReranker(llm_func=llm_func, **kwargs)
