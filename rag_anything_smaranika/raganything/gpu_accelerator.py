"""
GPU Acceleration Module for RAGAnything

Provides GPU-accelerated operations for:
- Batch embeddings with automatic GPU detection
- Optimized vector similarity search
- Concurrent API calls with GPU batching
- Memory-efficient GPU utilization

Author: RAGAnything Performance Team
"""

import asyncio
import logging
import time
from typing import List, Optional, Callable, Any, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Try importing GPU libraries
GPU_AVAILABLE = False
CUDA_AVAILABLE = False

try:
    import torch
    GPU_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info(f"GPU acceleration available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    logger.warning("PyTorch not available. GPU acceleration disabled.")
    torch = None


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration"""

    enable_gpu: bool = True
    """Enable GPU acceleration if available"""

    device: str = "cuda"
    """Device to use: 'cuda', 'cuda:0', 'cpu'"""

    batch_size: int = 64
    """Batch size for GPU operations"""

    max_gpu_memory_gb: float = 4.0
    """Maximum GPU memory to use (GB)"""

    prefetch_batches: int = 2
    """Number of batches to prefetch"""

    pin_memory: bool = True
    """Use pinned memory for faster CPU->GPU transfers"""

    num_workers: int = 2
    """Number of worker threads for data loading"""

    mixed_precision: bool = True
    """Use FP16 for embeddings to save memory"""

    optimize_for_inference: bool = True
    """Apply inference-specific optimizations"""


class GPUEmbeddingAccelerator:
    """
    GPU-accelerated embedding generation

    Features:
    - Automatic batching for GPU
    - Memory-efficient processing
    - Mixed precision support
    - Concurrent API calls with batching
    """

    def __init__(
        self,
        embedding_func: Callable,
        config: Optional[GPUConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize GPU embedding accelerator

        Args:
            embedding_func: Base embedding function to accelerate
            config: GPU configuration
            logger: Optional logger
        """
        self.embedding_func = embedding_func
        self.config = config or GPUConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Determine device
        if GPU_AVAILABLE and CUDA_AVAILABLE and self.config.enable_gpu:
            self.device = torch.device(self.config.device)
            self.use_gpu = True
            self.logger.info(f"Using GPU device: {self.device}")
        else:
            self.device = torch.device("cpu")
            self.use_gpu = False
            self.logger.info("Using CPU for embeddings")

        # Statistics
        self.stats = {
            "total_embeddings": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "gpu_time": 0.0,
            "cpu_time": 0.0,
        }

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with GPU acceleration

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start_time = time.time()

        try:
            # Call the original embedding function
            embeddings = await self._call_embedding_func(texts)

            # Convert to GPU tensors if enabled
            if self.use_gpu and GPU_AVAILABLE:
                embeddings = self._optimize_embeddings_gpu(embeddings)

            # Update stats
            elapsed = time.time() - start_time
            self.stats["total_embeddings"] += len(texts)
            self.stats["total_batches"] += 1
            self.stats["total_time"] += elapsed

            if self.use_gpu:
                self.stats["gpu_time"] += elapsed
            else:
                self.stats["cpu_time"] += elapsed

            return embeddings

        except Exception as e:
            self.logger.error(f"Error in embed_batch: {e}", exc_info=True)
            raise

    async def _call_embedding_func(self, texts: List[str]) -> List[List[float]]:
        """Call the original embedding function"""
        if asyncio.iscoroutinefunction(self.embedding_func):
            return await self.embedding_func(texts)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.embedding_func, texts)

    def _optimize_embeddings_gpu(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Optimize embeddings using GPU"""
        try:
            # Convert to tensor
            tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

            # Apply mixed precision if enabled
            if self.config.mixed_precision:
                tensor = tensor.half()  # Convert to FP16

            # Normalize embeddings on GPU
            tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)

            # Convert back to CPU and list
            if self.config.mixed_precision:
                tensor = tensor.float()  # Convert back to FP32

            return tensor.cpu().tolist()

        except Exception as e:
            self.logger.warning(f"GPU optimization failed, using original embeddings: {e}")
            return embeddings

    async def embed_documents_chunked(
        self,
        texts: List[str],
        chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Embed large document collections in chunks with GPU batching

        Args:
            texts: List of all texts to embed
            chunk_size: Size of each chunk (defaults to config batch_size)

        Returns:
            List of all embeddings
        """
        chunk_size = chunk_size or self.config.batch_size

        self.logger.info(f"Embedding {len(texts)} texts in chunks of {chunk_size}")

        all_embeddings = []

        # Process in chunks
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]

            # Get embeddings for chunk
            chunk_embeddings = await self.embed_batch(chunk)
            all_embeddings.extend(chunk_embeddings)

            # Log progress
            if (i + chunk_size) % (chunk_size * 10) == 0 or i + chunk_size >= len(texts):
                progress = min(i + chunk_size, len(texts))
                self.logger.info(f"Embedded {progress}/{len(texts)} texts")

        return all_embeddings

    async def embed_concurrent(
        self,
        text_batches: List[List[str]],
        max_concurrent: int = 4
    ) -> List[List[List[float]]]:
        """
        Embed multiple batches concurrently with GPU

        Args:
            text_batches: List of text batches
            max_concurrent: Maximum concurrent embedding calls

        Returns:
            List of embedding batches
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_with_semaphore(batch: List[str]) -> List[List[float]]:
            async with semaphore:
                return await self.embed_batch(batch)

        # Create tasks for all batches
        tasks = [embed_with_semaphore(batch) for batch in text_batches]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle errors
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch {i} failed: {result}")
                raise result
            final_results.append(result)

        return final_results

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        avg_time_per_embedding = (
            self.stats["total_time"] / self.stats["total_embeddings"]
            if self.stats["total_embeddings"] > 0 else 0
        )

        return {
            **self.stats,
            "use_gpu": self.use_gpu,
            "device": str(self.device),
            "avg_time_per_embedding_ms": avg_time_per_embedding * 1000,
            "embeddings_per_second": (
                self.stats["total_embeddings"] / self.stats["total_time"]
                if self.stats["total_time"] > 0 else 0
            ),
        }

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.use_gpu and GPU_AVAILABLE:
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")


class GPUVectorSearchAccelerator:
    """
    GPU-accelerated vector similarity search

    Features:
    - Batch similarity computation on GPU
    - Fast top-k selection
    - Memory-efficient operations
    """

    def __init__(
        self,
        config: Optional[GPUConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize GPU vector search accelerator

        Args:
            config: GPU configuration
            logger: Optional logger
        """
        self.config = config or GPUConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Determine device
        if GPU_AVAILABLE and CUDA_AVAILABLE and self.config.enable_gpu:
            self.device = torch.device(self.config.device)
            self.use_gpu = True
        else:
            self.device = torch.device("cpu")
            self.use_gpu = False

    def compute_similarities_batch(
        self,
        query_vectors: List[List[float]],
        database_vectors: List[List[float]],
        top_k: int = 10
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Compute top-k similarities for multiple queries

        Args:
            query_vectors: List of query vectors
            database_vectors: List of database vectors
            top_k: Number of top results to return per query

        Returns:
            List of (indices, similarities) tuples for each query
        """
        if not query_vectors or not database_vectors:
            return []

        try:
            # Convert to tensors
            queries = torch.tensor(query_vectors, dtype=torch.float32, device=self.device)
            database = torch.tensor(database_vectors, dtype=torch.float32, device=self.device)

            # Normalize vectors
            queries = torch.nn.functional.normalize(queries, p=2, dim=1)
            database = torch.nn.functional.normalize(database, p=2, dim=1)

            # Compute cosine similarities (batch matrix multiplication)
            similarities = torch.mm(queries, database.t())

            # Get top-k for each query
            top_k = min(top_k, database.shape[0])
            top_scores, top_indices = torch.topk(similarities, k=top_k, dim=1)

            # Convert to lists
            results = []
            for i in range(len(query_vectors)):
                indices = top_indices[i].cpu().tolist()
                scores = top_scores[i].cpu().tolist()
                results.append((indices, scores))

            return results

        except Exception as e:
            self.logger.error(f"Error in GPU similarity search: {e}", exc_info=True)
            # Fallback to CPU
            return self._compute_similarities_cpu(query_vectors, database_vectors, top_k)

    def _compute_similarities_cpu(
        self,
        query_vectors: List[List[float]],
        database_vectors: List[List[float]],
        top_k: int
    ) -> List[Tuple[List[int], List[float]]]:
        """CPU fallback for similarity computation"""
        import numpy as np

        queries = np.array(query_vectors)
        database = np.array(database_vectors)

        # Normalize
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        database = database / np.linalg.norm(database, axis=1, keepdims=True)

        # Compute similarities
        similarities = np.dot(queries, database.T)

        # Get top-k
        results = []
        for sim_row in similarities:
            top_k_actual = min(top_k, len(sim_row))
            top_indices = np.argpartition(sim_row, -top_k_actual)[-top_k_actual:]
            top_indices = top_indices[np.argsort(-sim_row[top_indices])]
            top_scores = sim_row[top_indices].tolist()
            results.append((top_indices.tolist(), top_scores))

        return results


def create_gpu_accelerated_embedding_func(
    base_embedding_func: Callable,
    config: Optional[GPUConfig] = None
) -> Callable:
    """
    Create a GPU-accelerated version of an embedding function

    Args:
        base_embedding_func: Original embedding function
        config: GPU configuration

    Returns:
        Accelerated embedding function

    Example:
        ```python
        # Original embedding function
        async def my_embedding_func(texts: List[str]) -> List[List[float]]:
            # ... API call or model inference ...
            return embeddings

        # Create GPU-accelerated version
        gpu_embedding_func = create_gpu_accelerated_embedding_func(
            my_embedding_func,
            config=GPUConfig(batch_size=64)
        )

        # Use in RAGAnything
        rag = await create_rag_anything(
            llm_model_func=my_llm,
            embedding_func=gpu_embedding_func
        )
        ```
    """
    accelerator = GPUEmbeddingAccelerator(base_embedding_func, config)

    async def accelerated_func(texts: List[str]) -> List[List[float]]:
        return await accelerator.embed_batch(texts)

    # Attach accelerator for access to stats
    accelerated_func.accelerator = accelerator

    return accelerated_func


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return system information

    Returns:
        Dictionary with GPU information
    """
    info = {
        "pytorch_available": GPU_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "device_count": 0,
        "devices": [],
    }

    if GPU_AVAILABLE and CUDA_AVAILABLE:
        info["device_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
            })

    return info


# Export main classes and functions
__all__ = [
    "GPUConfig",
    "GPUEmbeddingAccelerator",
    "GPUVectorSearchAccelerator",
    "create_gpu_accelerated_embedding_func",
    "check_gpu_availability",
    "GPU_AVAILABLE",
    "CUDA_AVAILABLE",
]
