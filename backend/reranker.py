"""
Reranking Module for RAG-Anything

Provides reranking functionality using:
1. Gemini-based LLM reranking (free tier compatible)
2. Cross-encoder style scoring
3. Relevance-based reordering

Reranking is crucial for RAG systems because:
- Vector search (embeddings) finds semantically similar text but may miss subtle context
- LLMs can deeply understand query intent and document relevance
- Reranking improves answer quality by promoting truly relevant chunks to the top

Author: RAG-Anything Team
Version: 1.0.0
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class GeminiReranker:
    """
    Reranker using Gemini API for semantic relevance scoring

    This reranker takes chunks from vector search and re-scores them
    based on deep semantic understanding using an LLM.

    Why reranking matters:
    ---------------------
    Vector embeddings alone can miss:
    - Negations ("not effective" vs "effective")
    - Context dependencies ("aspirin for elderly" vs "aspirin for children")
    - Query intent ("what causes X" vs "how to prevent X")

    LLM reranking provides:
    - Contextual understanding of the query
    - Semantic relevance beyond keyword matching
    - Better handling of complex queries
    """

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
            model_name: Gemini model to use (default: flash for speed)
            batch_size: Number of chunks to process in parallel
            temperature: Temperature for relevance scoring (low=consistent)
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

        Process:
        1. Take top chunks from vector search (e.g., top 50)
        2. Score each chunk's relevance using LLM (0-10 scale)
        3. Re-order by relevance score
        4. Return top_k most relevant chunks

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
            # Score all chunks in batches
            scored_chunks = await self._score_chunks_batch(query, chunks)

            # Sort by relevance score (highest first)
            scored_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            # Return top_k if specified
            if top_k:
                scored_chunks = scored_chunks[:top_k]

            logger.info(
                f"Reranking complete. Top score: {scored_chunks[0].get('relevance_score', 0):.2f}, "
                f"Bottom score: {scored_chunks[-1].get('relevance_score', 0):.2f}"
            )

            return scored_chunks

        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            # Return original order on error
            return chunks[:top_k] if top_k else chunks

    async def _score_chunks_batch(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score chunks in batches for efficiency

        Args:
            query: Search query
            chunks: List of chunks to score

        Returns:
            Chunks with relevance_score added
        """
        scored_chunks = []

        # Process in batches to avoid rate limits
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]

            # Score batch concurrently
            tasks = [self._score_chunk(query, chunk) for chunk in batch]
            batch_scores = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for chunk, score_result in zip(batch, batch_scores):
                if isinstance(score_result, Exception):
                    logger.warning(f"Failed to score chunk: {score_result}")
                    chunk['relevance_score'] = 0.0
                else:
                    chunk['relevance_score'] = score_result

                scored_chunks.append(chunk)

        return scored_chunks

    async def _score_chunk(
        self,
        query: str,
        chunk: Dict[str, Any]
    ) -> float:
        """
        Score a single chunk's relevance to the query using LLM

        Prompt engineering approach:
        - Ask LLM to act as a relevance judge
        - Provide clear scoring criteria (0-10 scale)
        - Extract numeric score from response

        Args:
            query: Search query
            chunk: Chunk dictionary with 'content' field

        Returns:
            Relevance score (0-10)
        """
        content = chunk.get('content', '')
        if not content:
            return 0.0

        # Truncate very long chunks to avoid token limits
        max_content_length = 1000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        # Prompt for relevance scoring
        prompt = f"""You are a relevance judge. Score how relevant the following passage is to answering the query.

Query: {query}

Passage:
{content}

Scoring criteria:
10 = Directly answers the query with specific, relevant information
8-9 = Highly relevant, provides useful context
6-7 = Somewhat relevant, contains related information
4-5 = Tangentially related, limited usefulness
2-3 = Barely related, mostly off-topic
0-1 = Completely irrelevant

Respond with ONLY a number from 0-10. No explanation needed."""

        try:
            # Call LLM for scoring
            if self.llm_func:
                response = await self.llm_func(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=50  # Increased from 10 to allow for complete score responses
                )
            else:
                # Fallback: no reranking
                return 5.0

            # Extract numeric score from response
            score = self._extract_score(response)
            return score

        except Exception as e:
            logger.error(f"Error scoring chunk: {e}")
            return 5.0  # Default mid-range score on error

    def _extract_score(self, response: str) -> float:
        """
        Extract numeric score from LLM response

        Handles various response formats:
        - "8.5"
        - "Score: 9"
        - "The relevance is 7/10"
        - "8"

        Args:
            response: LLM response text

        Returns:
            Extracted score (0-10), defaults to 5.0 if parsing fails
        """
        try:
            # Remove whitespace
            response = response.strip()

            # Try to find a number (int or float) in the response
            # Pattern matches: "8", "8.5", "9/10", "Score: 7", etc.
            number_pattern = r'(\d+\.?\d*)'
            matches = re.findall(number_pattern, response)

            if matches:
                # Take the first number found
                score = float(matches[0])

                # Normalize to 0-10 range
                score = max(0.0, min(10.0, score))

                return score
            else:
                logger.warning(f"Could not extract score from response: {response}")
                return 5.0

        except Exception as e:
            logger.error(f"Error extracting score: {e}")
            return 5.0


# Example usage
async def main():
    """Example demonstrating reranking"""
    # Mock LLM function for testing
    async def mock_llm(prompt: str, **kwargs) -> str:
        # Simulate scoring based on keyword matching
        if "directly" in prompt.lower():
            return "9"
        elif "somewhat" in prompt.lower():
            return "6"
        else:
            return "3"

    # Create reranker
    reranker = GeminiReranker(llm_func=mock_llm)

    # Example query and chunks
    query = "What are the side effects of aspirin?"

    chunks = [
        {"content": "Aspirin can cause stomach bleeding in some patients..."},
        {"content": "The history of aspirin dates back to ancient times..."},
        {"content": "Common side effects include nausea and heartburn..."},
    ]

    # Rerank
    reranked = await reranker.rerank(query, chunks, top_k=2)

    print("Reranked results:")
    for i, chunk in enumerate(reranked, 1):
        print(f"{i}. Score: {chunk['relevance_score']:.1f} - {chunk['content'][:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
