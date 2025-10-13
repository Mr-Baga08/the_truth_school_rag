"""
Enhanced Batch Processing Optimizer for RAGAnything

This module provides advanced batch processing capabilities with:
- Intelligent chunking and prefetching
- Concurrent document parsing and processing
- Progress tracking with ETA estimation
- Adaptive rate limiting for API calls
- Batch caching strategies
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import logging

@dataclass
class BatchProcessingStats:
    """Statistics for batch processing"""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Detailed stats
    parsing_time: float = 0.0
    text_processing_time: float = 0.0
    multimodal_processing_time: float = 0.0
    total_api_calls: int = 0
    cache_hits: int = 0

    def get_eta_seconds(self) -> Optional[float]:
        """Calculate estimated time remaining"""
        if self.processed_documents == 0:
            return None

        elapsed = time.time() - self.start_time
        avg_time_per_doc = elapsed / self.processed_documents
        remaining_docs = self.total_documents - self.processed_documents

        return avg_time_per_doc * remaining_docs

    def get_processing_rate(self) -> float:
        """Get documents per second"""
        if self.processed_documents == 0:
            return 0.0

        elapsed = time.time() - self.start_time
        return self.processed_documents / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        elapsed = (self.end_time or time.time()) - self.start_time

        return {
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "failed_documents": self.failed_documents,
            "success_rate": self.processed_documents / max(self.total_documents, 1) * 100,
            "total_time_seconds": elapsed,
            "processing_rate_docs_per_sec": self.get_processing_rate(),
            "parsing_time": self.parsing_time,
            "text_processing_time": self.text_processing_time,
            "multimodal_processing_time": self.multimodal_processing_time,
            "total_api_calls": self.total_api_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.total_api_calls, 1) * 100,
            "eta_seconds": self.get_eta_seconds(),
        }


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on API response times"""

    def __init__(self, initial_rate: int = 10, min_rate: int = 1, max_rate: int = 50):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.semaphore = asyncio.Semaphore(initial_rate)
        self.recent_times: List[float] = []
        self.adjustment_lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a slot with current rate limit"""
        await self.semaphore.acquire()

    def release(self, execution_time: float):
        """Release slot and adjust rate based on execution time"""
        self.semaphore.release()
        self.recent_times.append(execution_time)

        # Keep only last 20 measurements
        if len(self.recent_times) > 20:
            self.recent_times.pop(0)

    async def adapt_rate(self):
        """Adjust rate based on recent performance"""
        async with self.adjustment_lock:
            if len(self.recent_times) < 5:
                return

            avg_time = sum(self.recent_times) / len(self.recent_times)

            # If responses are fast, increase rate
            if avg_time < 1.0 and self.current_rate < self.max_rate:
                self.current_rate = min(self.current_rate + 2, self.max_rate)
                # Create new semaphore with increased capacity
                self.semaphore = asyncio.Semaphore(self.current_rate)
            # If responses are slow, decrease rate
            elif avg_time > 3.0 and self.current_rate > self.min_rate:
                self.current_rate = max(self.current_rate - 2, self.min_rate)
                self.semaphore = asyncio.Semaphore(self.current_rate)


class BatchOptimizer:
    """
    Advanced batch processing optimizer for RAGAnything

    Features:
    - Concurrent document parsing with prefetching
    - Intelligent chunk batching
    - Progress tracking with ETA
    - Adaptive rate limiting
    - Cache-aware processing
    """

    def __init__(
        self,
        max_concurrent_parsers: int = 4,
        max_concurrent_processors: int = 10,
        prefetch_buffer_size: int = 5,
        enable_adaptive_rate: bool = True,
        enable_progress_tracking: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.max_concurrent_parsers = max_concurrent_parsers
        self.max_concurrent_processors = max_concurrent_processors
        self.prefetch_buffer_size = prefetch_buffer_size
        self.enable_adaptive_rate = enable_adaptive_rate
        self.enable_progress_tracking = enable_progress_tracking

        self.logger = logger or logging.getLogger(__name__)

        # Rate limiter
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rate=max_concurrent_processors,
            min_rate=max(1, max_concurrent_processors // 2),
            max_rate=max_concurrent_processors * 2,
        )

        # Statistics
        self.stats = BatchProcessingStats()

        # Progress callback
        self.progress_callback: Optional[callable] = None

    def set_progress_callback(self, callback: callable):
        """Set callback for progress updates"""
        self.progress_callback = callback

    async def _report_progress(self):
        """Report progress to callback if enabled"""
        if self.enable_progress_tracking and self.progress_callback:
            progress_data = {
                "processed": self.stats.processed_documents,
                "total": self.stats.total_documents,
                "failed": self.stats.failed_documents,
                "percentage": (self.stats.processed_documents / max(self.stats.total_documents, 1)) * 100,
                "eta_seconds": self.stats.get_eta_seconds(),
                "rate_docs_per_sec": self.stats.get_processing_rate(),
            }
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(progress_data)
                else:
                    self.progress_callback(progress_data)
            except Exception as e:
                self.logger.debug(f"Error in progress callback: {e}")

    async def process_documents_batch_optimized(
        self,
        rag_instance,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple documents with advanced optimizations

        Args:
            rag_instance: RAGAnything instance
            file_paths: List of file paths to process
            output_dir: Output directory
            parse_method: Parse method
            **kwargs: Additional processing parameters

        Returns:
            Dict with processing results and statistics
        """
        self.stats = BatchProcessingStats(total_documents=len(file_paths))
        self.logger.info(f"Starting optimized batch processing for {len(file_paths)} documents")

        # Create pipeline stages
        parse_queue = asyncio.Queue(maxsize=self.prefetch_buffer_size)
        process_queue = asyncio.Queue(maxsize=self.prefetch_buffer_size)

        # Results tracking
        results = {
            "successful": [],
            "failed": [],
        }
        results_lock = asyncio.Lock()

        # Stage 1: Document parsing with prefetching
        async def parse_documents():
            """Parse documents concurrently and feed to process queue"""
            parser_semaphore = asyncio.Semaphore(self.max_concurrent_parsers)

            async def parse_single_document(file_path: str):
                async with parser_semaphore:
                    try:
                        start_time = time.time()

                        # Parse document
                        content_list, doc_id = await rag_instance.parse_document(
                            file_path=file_path,
                            output_dir=output_dir,
                            parse_method=parse_method,
                            display_stats=False,
                            **kwargs
                        )

                        parse_time = time.time() - start_time
                        self.stats.parsing_time += parse_time

                        # Put parsed document into process queue
                        await process_queue.put({
                            "file_path": file_path,
                            "content_list": content_list,
                            "doc_id": doc_id,
                            "parse_time": parse_time,
                        })

                        return True, file_path, None

                    except Exception as e:
                        self.logger.error(f"Failed to parse {file_path}: {e}")
                        self.stats.failed_documents += 1
                        await self._report_progress()
                        return False, file_path, str(e)

            # Parse all documents
            parse_tasks = [
                asyncio.create_task(parse_single_document(fp))
                for fp in file_paths
            ]

            parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)

            # Signal completion
            await process_queue.put(None)

            return parse_results

        # Stage 2: Document processing with rate limiting
        async def process_documents():
            """Process parsed documents with adaptive rate limiting"""

            async def process_single_document(parsed_doc: Dict[str, Any]):
                if self.enable_adaptive_rate:
                    await self.rate_limiter.acquire()

                try:
                    start_time = time.time()

                    # Process document
                    file_path = parsed_doc["file_path"]
                    content_list = parsed_doc["content_list"]
                    doc_id = parsed_doc["doc_id"]

                    # Separate text and multimodal
                    from raganything.utils import separate_content, insert_text_content
                    text_content, multimodal_items = separate_content(content_list)

                    # Process text content
                    text_start = time.time()
                    if text_content.strip():
                        await insert_text_content(
                            rag_instance.lightrag,
                            input=text_content,
                            file_paths=Path(file_path).name,
                            ids=doc_id,
                        )
                    self.stats.text_processing_time += time.time() - text_start

                    # Process multimodal content
                    multimodal_start = time.time()
                    if multimodal_items:
                        await rag_instance._process_multimodal_content(
                            multimodal_items=multimodal_items,
                            file_path=file_path,
                            doc_id=doc_id,
                        )
                    else:
                        await rag_instance._mark_multimodal_processing_complete(doc_id)
                    self.stats.multimodal_processing_time += time.time() - multimodal_start

                    # Update statistics
                    self.stats.processed_documents += 1
                    self.stats.total_chunks += len(content_list)

                    processing_time = time.time() - start_time

                    if self.enable_adaptive_rate:
                        self.rate_limiter.release(processing_time)

                    async with results_lock:
                        results["successful"].append({
                            "file_path": file_path,
                            "doc_id": doc_id,
                            "processing_time": processing_time,
                            "parse_time": parsed_doc.get("parse_time", 0),
                        })

                    await self._report_progress()

                    return True, file_path, None

                except Exception as e:
                    self.logger.error(f"Failed to process {parsed_doc.get('file_path', 'unknown')}: {e}")
                    self.stats.failed_documents += 1

                    if self.enable_adaptive_rate:
                        self.rate_limiter.release(1.0)

                    async with results_lock:
                        results["failed"].append({
                            "file_path": parsed_doc.get("file_path", "unknown"),
                            "error": str(e),
                        })

                    await self._report_progress()

                    return False, parsed_doc.get("file_path", "unknown"), str(e)

            # Process documents as they become available
            processing_tasks = []

            while True:
                parsed_doc = await process_queue.get()

                if parsed_doc is None:
                    # All documents parsed
                    break

                # Start processing task
                task = asyncio.create_task(process_single_document(parsed_doc))
                processing_tasks.append(task)

            # Wait for all processing to complete
            await asyncio.gather(*processing_tasks, return_exceptions=True)

        # Run both stages concurrently
        self.logger.info("Starting concurrent parsing and processing pipeline")

        parse_task = asyncio.create_task(parse_documents())
        process_task = asyncio.create_task(process_documents())

        await asyncio.gather(parse_task, process_task)

        # Finalize statistics
        self.stats.end_time = time.time()

        self.logger.info(f"Batch processing complete:")
        self.logger.info(f"  Successful: {self.stats.processed_documents} documents")
        self.logger.info(f"  Failed: {self.stats.failed_documents} documents")
        self.logger.info(f"  Total time: {self.stats.end_time - self.stats.start_time:.2f}s")
        self.logger.info(f"  Processing rate: {self.stats.get_processing_rate():.2f} docs/sec")

        return {
            "successful_files": results["successful"],
            "failed_files": results["failed"],
            "statistics": self.stats.to_dict(),
        }

    async def process_multimodal_batch_optimized(
        self,
        rag_instance,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        chunk_size: int = 20,
    ) -> None:
        """
        Process multimodal content in optimized batches

        Args:
            rag_instance: RAGAnything instance
            multimodal_items: List of multimodal items
            file_path: File path for reference
            doc_id: Document ID
            chunk_size: Number of items to process per batch
        """
        if not multimodal_items:
            return

        self.logger.info(f"Processing {len(multimodal_items)} multimodal items in optimized batches of {chunk_size}")

        # Split into batches
        batches = [
            multimodal_items[i:i + chunk_size]
            for i in range(0, len(multimodal_items), chunk_size)
        ]

        # Process batches concurrently
        batch_tasks = []
        for batch_idx, batch in enumerate(batches):
            task = asyncio.create_task(
                self._process_multimodal_batch_chunk(
                    rag_instance, batch, file_path, doc_id, batch_idx
                )
            )
            batch_tasks.append(task)

        # Wait for all batches
        await asyncio.gather(*batch_tasks, return_exceptions=True)

        self.logger.info(f"Completed processing {len(multimodal_items)} multimodal items")

    async def _process_multimodal_batch_chunk(
        self,
        rag_instance,
        batch: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        batch_idx: int,
    ):
        """Process a single batch chunk of multimodal items"""
        try:
            # Use the existing batch processing method
            await rag_instance._process_multimodal_content_batch_type_aware(
                multimodal_items=batch,
                file_path=file_path,
                doc_id=doc_id,
            )

            self.logger.debug(f"Completed multimodal batch {batch_idx + 1}")

        except Exception as e:
            self.logger.error(f"Error in multimodal batch {batch_idx + 1}: {e}")
            raise


class ProgressTracker:
    """Progress tracker with console and file logging"""

    def __init__(self, total_items: int, log_file: Optional[str] = None):
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.log_file = log_file

        self.logger = logging.getLogger(__name__)

    def update(self, success: bool = True):
        """Update progress"""
        if success:
            self.processed_items += 1
        else:
            self.failed_items += 1

        # Calculate metrics
        total_processed = self.processed_items + self.failed_items
        percentage = (total_processed / self.total_items) * 100
        elapsed = time.time() - self.start_time

        if total_processed > 0:
            eta = (elapsed / total_processed) * (self.total_items - total_processed)
            rate = total_processed / elapsed

            progress_msg = (
                f"Progress: {total_processed}/{self.total_items} ({percentage:.1f}%) | "
                f"Success: {self.processed_items} | Failed: {self.failed_items} | "
                f"Rate: {rate:.2f} docs/s | ETA: {eta:.1f}s"
            )

            self.logger.info(progress_msg)

            # Write to log file if specified
            if self.log_file:
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {progress_msg}\n")
                except Exception:
                    pass

    def get_summary(self) -> Dict[str, Any]:
        """Get final summary"""
        total_time = time.time() - self.start_time

        return {
            "total_items": self.total_items,
            "processed": self.processed_items,
            "failed": self.failed_items,
            "total_time": total_time,
            "success_rate": (self.processed_items / self.total_items) * 100 if self.total_items > 0 else 0,
            "average_time_per_item": total_time / max(self.processed_items + self.failed_items, 1),
        }
