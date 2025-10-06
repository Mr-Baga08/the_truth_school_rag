"""
Web Search Module for RAG-Anything using Tavily API

Provides intelligent web search capabilities to augment RAG with real-time information.

Features:
- Tavily API integration for high-quality search results
- Context-aware search query generation
- Result filtering and ranking
- Hybrid RAG + Web search mode

Author: RAG-Anything Team
Version: 1.0.0
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient, AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily not installed. Install with: pip install tavily-python")


class WebSearcher:
    """Web search integration using Tavily API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_raw_content: bool = True
    ):
        """
        Initialize web searcher

        Args:
            api_key: Tavily API key (from env if not provided)
            max_results: Maximum number of search results to return
            search_depth: "basic" or "advanced" (advanced is more thorough)
            include_raw_content: Whether to include full page content
        """
        if not TAVILY_AVAILABLE:
            raise ImportError("Tavily is not installed. Install with: pip install tavily-python")

        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key not found. Set TAVILY_API_KEY environment variable.")

        self.max_results = max_results
        self.search_depth = search_depth
        self.include_raw_content = include_raw_content

        # Initialize async client
        self.client = AsyncTavilyClient(api_key=self.api_key)

        logger.info(f"WebSearcher initialized (max_results={max_results}, depth={search_depth})")

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform web search

        Args:
            query: Search query
            max_results: Override default max results
            include_domains: Only search these domains
            exclude_domains: Exclude these domains
            search_depth: Override default search depth

        Returns:
            Dictionary with search results and metadata
        """
        try:
            logger.info(f"Searching web: {query[:100]}...")

            # Build search parameters
            search_params = {
                "query": query,
                "max_results": max_results or self.max_results,
                "search_depth": search_depth or self.search_depth,
                "include_raw_content": self.include_raw_content,
            }

            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            # Perform search
            response = await self.client.search(**search_params)

            # Process results
            results = {
                "query": query,
                "results": response.get("results", []),
                "answer": response.get("answer", ""),  # Tavily's AI-generated answer
                "search_metadata": {
                    "total_results": len(response.get("results", [])),
                    "search_depth": search_params["search_depth"],
                    "timestamp": datetime.now().isoformat(),
                }
            }

            logger.info(f"Web search complete: {len(results['results'])} results found")
            return results

        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return {
                "query": query,
                "results": [],
                "answer": "",
                "error": str(e),
                "search_metadata": {
                    "total_results": 0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            }

    async def search_with_context(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search with additional context to refine query

        Args:
            query: Base search query
            context: Additional context to help refine search
            **kwargs: Additional search parameters

        Returns:
            Search results dictionary
        """
        # If context provided, enhance query
        if context:
            enhanced_query = f"{query} {context}"
        else:
            enhanced_query = query

        return await self.search(enhanced_query, **kwargs)

    def format_results_for_rag(self, search_results: Dict[str, Any]) -> str:
        """
        Format web search results for RAG context

        Args:
            search_results: Results from search()

        Returns:
            Formatted string for RAG context
        """
        if not search_results.get("results"):
            return "No web search results found."

        formatted = ["=== Web Search Results ===\n"]

        # Add Tavily's answer if available
        if search_results.get("answer"):
            formatted.append(f"Quick Answer: {search_results['answer']}\n")

        # Add individual results
        for idx, result in enumerate(search_results["results"], 1):
            formatted.append(f"\n[Source {idx}] {result.get('title', 'Untitled')}")
            formatted.append(f"URL: {result.get('url', 'N/A')}")
            formatted.append(f"Content: {result.get('content', 'No content')[:500]}...")
            if result.get("score"):
                formatted.append(f"Relevance: {result['score']:.2f}")

        formatted.append(f"\n=== End of Web Results ({len(search_results['results'])} sources) ===")
        return "\n".join(formatted)

    def format_results_for_llm(self, search_results: Dict[str, Any]) -> str:
        """
        Format web search results optimally for LLM processing

        Args:
            search_results: Results from search()

        Returns:
            Structured string optimized for LLM comprehension
        """
        if not search_results.get("results"):
            return "No relevant web search results were found for this query."

        formatted = []

        # Add Tavily's AI-generated answer first (if available)
        if search_results.get("answer"):
            formatted.append("### AI-Generated Summary:")
            formatted.append(search_results['answer'])
            formatted.append("")

        # Add detailed source information
        formatted.append("### Detailed Sources:")
        formatted.append("")

        for idx, result in enumerate(search_results["results"], 1):
            formatted.append(f"**Source {idx}: {result.get('title', 'Untitled')}**")
            formatted.append(f"- URL: {result.get('url', 'N/A')}")
            formatted.append(f"- Published: {result.get('published_date', 'Unknown date')}")

            # Get content (full or truncated based on availability)
            content = result.get('content', '')
            if result.get('raw_content') and len(result.get('raw_content', '')) > len(content):
                content = result['raw_content'][:2000]  # Use more detailed content

            formatted.append(f"- Content: {content}")

            if result.get("score"):
                formatted.append(f"- Relevance Score: {result['score']:.2%}")

            formatted.append("")

        formatted.append(f"*Total sources: {len(search_results['results'])}*")
        return "\n".join(formatted)

    async def hybrid_search(
        self,
        query: str,
        rag_results: Optional[str] = None,
        combine_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Hybrid search: combine RAG results with web search

        Args:
            query: Search query
            rag_results: Results from RAG system
            combine_results: Whether to combine RAG and web results
            **kwargs: Additional search parameters

        Returns:
            Dictionary with combined results
        """
        # Perform web search
        web_results = await self.search(query, **kwargs)

        if not combine_results:
            return web_results

        # Combine RAG and web results
        combined_context = []

        if rag_results:
            combined_context.append("=== Knowledge Base Results ===")
            combined_context.append(rag_results)
            combined_context.append("")

        combined_context.append(self.format_results_for_rag(web_results))

        return {
            "query": query,
            "combined_context": "\n".join(combined_context),
            "rag_results": rag_results,
            "web_results": web_results,
            "metadata": {
                "has_rag_results": bool(rag_results),
                "has_web_results": len(web_results.get("results", [])) > 0,
                "timestamp": datetime.now().isoformat(),
            }
        }


def create_web_searcher(api_key: Optional[str] = None, **kwargs) -> WebSearcher:
    """
    Factory function to create a web searcher

    Args:
        api_key: Tavily API key
        **kwargs: Additional WebSearcher parameters

    Returns:
        Configured WebSearcher instance
    """
    return WebSearcher(api_key=api_key, **kwargs)
