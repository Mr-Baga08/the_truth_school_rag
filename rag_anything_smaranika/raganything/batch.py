"""
Batch processing functionality for RAGAnything

Contains methods for processing multiple documents in batch mode
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Callable
import time

from .batch_parser import BatchParser, BatchProcessingResult
from .batch_optimizer import BatchOptimizer, ProgressTracker

if TYPE_CHECKING:
    from .config import RAGAnythingConfig


class BatchMixin:
    """BatchMixin class containing batch processing functionality for RAGAnything"""

    # Type hints for mixin attributes (will be available when mixed into RAGAnything)
    config: "RAGAnythingConfig"
    logger: logging.Logger

    # Type hints for methods that will be available from other mixins
    async def _ensure_lightrag_initialized(self) -> None: ...
    async def process_document_complete(self, file_path: str, **kwargs) -> None: ...

    # ==========================================
    # ORIGINAL BATCH PROCESSING METHOD (RESTORED)
    # ==========================================

    async def process_folder_complete(
        self,
        folder_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = None,
        max_workers: int = None,
    ):
        """
        Process all supported files in a folder

        Args:
            folder_path: Path to the folder containing files to process
            output_dir: Directory for parsed outputs (optional)
            parse_method: Parsing method to use (optional)
            display_stats: Whether to display statistics (optional)
            split_by_character: Character to split by (optional)
            split_by_character_only: Whether to split only by character (optional)
            file_extensions: List of file extensions to process (optional)
            recursive: Whether to process folders recursively (optional)
            max_workers: Maximum number of workers for concurrent processing (optional)
        """
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = True
        if file_extensions is None:
            file_extensions = self.config.supported_file_extensions
        if recursive is None:
            recursive = self.config.recursive_folder_processing
        if max_workers is None:
            max_workers = self.config.max_concurrent_files

        await self._ensure_lightrag_initialized()

        # Get all files in the folder
        folder_path_obj = Path(folder_path)
        if not folder_path_obj.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Collect files based on supported extensions
        files_to_process = []
        for file_ext in file_extensions:
            if recursive:
                pattern = f"**/*{file_ext}"
            else:
                pattern = f"*{file_ext}"
            files_to_process.extend(folder_path_obj.glob(pattern))

        if not files_to_process:
            self.logger.warning(f"No supported files found in {folder_path}")
            return

        self.logger.info(
            f"Found {len(files_to_process)} files to process in {folder_path}"
        )

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process files with controlled concurrency
        semaphore = asyncio.Semaphore(max_workers)
        tasks = []

        async def process_single_file(file_path: Path):
            async with semaphore:
                try:
                    await self.process_document_complete(
                        str(file_path),
                        output_dir=output_dir,
                        parse_method=parse_method,
                        split_by_character=split_by_character,
                        split_by_character_only=split_by_character_only,
                    )
                    return True, str(file_path), None
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
                    return False, str(file_path), str(e)

        # Create tasks for all files
        for file_path in files_to_process:
            task = asyncio.create_task(process_single_file(file_path))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_files = []
        failed_files = []
        for result in results:
            if isinstance(result, Exception):
                failed_files.append(("unknown", str(result)))
            else:
                success, file_path, error = result
                if success:
                    successful_files.append(file_path)
                else:
                    failed_files.append((file_path, error))

        # Display statistics if requested
        if display_stats:
            self.logger.info("Processing complete!")
            self.logger.info(f"  Successful: {len(successful_files)} files")
            self.logger.info(f"  Failed: {len(failed_files)} files")
            if failed_files:
                self.logger.warning("Failed files:")
                for file_path, error in failed_files:
                    self.logger.warning(f"  - {file_path}: {error}")

    # ==========================================
    # NEW ENHANCED BATCH PROCESSING METHODS
    # ==========================================

    def process_documents_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> BatchProcessingResult:
        """
        Process multiple documents in batch using the new BatchParser

        Args:
            file_paths: List of file paths or directories to process
            output_dir: Output directory for parsed files
            parse_method: Parsing method to use
            max_workers: Maximum number of workers for parallel processing
            recursive: Whether to process directories recursively
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments passed to the parser

        Returns:
            BatchProcessingResult: Results of the batch processing
        """
        # Use config defaults if not specified
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        # Create batch parser
        batch_parser = BatchParser(
            parser_type=self.config.parser,
            max_workers=max_workers,
            show_progress=show_progress,
            skip_installation_check=True,  # Skip installation check for better UX
        )

        # Process batch
        return batch_parser.process_batch(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            recursive=recursive,
            **kwargs,
        )

    async def process_documents_batch_async(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> BatchProcessingResult:
        """
        Asynchronously process multiple documents in batch

        Args:
            file_paths: List of file paths or directories to process
            output_dir: Output directory for parsed files
            parse_method: Parsing method to use
            max_workers: Maximum number of workers for parallel processing
            recursive: Whether to process directories recursively
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments passed to the parser

        Returns:
            BatchProcessingResult: Results of the batch processing
        """
        # Use config defaults if not specified
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        # Create batch parser
        batch_parser = BatchParser(
            parser_type=self.config.parser,
            max_workers=max_workers,
            show_progress=show_progress,
            skip_installation_check=True,  # Skip installation check for better UX
        )

        # Process batch asynchronously
        return await batch_parser.process_batch_async(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            recursive=recursive,
            **kwargs,
        )

    def get_supported_file_extensions(self) -> List[str]:
        """Get list of supported file extensions for batch processing"""
        batch_parser = BatchParser(parser_type=self.config.parser)
        return batch_parser.get_supported_extensions()

    def filter_supported_files(
        self, file_paths: List[str], recursive: Optional[bool] = None
    ) -> List[str]:
        """
        Filter file paths to only include supported file types

        Args:
            file_paths: List of file paths to filter
            recursive: Whether to process directories recursively

        Returns:
            List of supported file paths
        """
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        batch_parser = BatchParser(parser_type=self.config.parser)
        return batch_parser.filter_supported_files(file_paths, recursive)

    async def process_documents_with_rag_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_workers: Optional[int] = None,
        recursive: Optional[bool] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process documents in batch and then add them to RAG

        This method combines document parsing and RAG insertion:
        1. First, parse all documents using batch processing
        2. Then, process each successfully parsed document with RAG

        Args:
            file_paths: List of file paths or directories to process
            output_dir: Output directory for parsed files
            parse_method: Parsing method to use
            max_workers: Maximum number of workers for parallel processing
            recursive: Whether to process directories recursively
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments passed to the parser

        Returns:
            Dict containing both parse results and RAG processing results
        """
        start_time = time.time()

        # Use config defaults if not specified
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if max_workers is None:
            max_workers = self.config.max_concurrent_files
        if recursive is None:
            recursive = self.config.recursive_folder_processing

        self.logger.info("Starting batch processing with RAG integration")

        # Step 1: Parse documents in batch
        parse_result = self.process_documents_batch(
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            max_workers=max_workers,
            recursive=recursive,
            show_progress=show_progress,
            **kwargs,
        )

        # Step 2: Process with RAG
        # Initialize RAG system
        await self._ensure_lightrag_initialized()

        # Then, process each successful file with RAG
        rag_results = {}

        if parse_result.successful_files:
            self.logger.info(
                f"Processing {len(parse_result.successful_files)} files with RAG"
            )

            # Process files with RAG (this could be parallelized in the future)
            for file_path in parse_result.successful_files:
                try:
                    # Process the successfully parsed file with RAG
                    await self.process_document_complete(
                        file_path,
                        output_dir=output_dir,
                        parse_method=parse_method,
                        **kwargs,
                    )

                    # Get some statistics about the processed content
                    # This would require additional tracking in the RAG system
                    rag_results[file_path] = {"status": "success", "processed": True}

                except Exception as e:
                    self.logger.error(
                        f"Failed to process {file_path} with RAG: {str(e)}"
                    )
                    rag_results[file_path] = {
                        "status": "failed",
                        "error": str(e),
                        "processed": False,
                    }

        processing_time = time.time() - start_time

        return {
            "parse_result": parse_result,
            "rag_results": rag_results,
            "total_processing_time": processing_time,
            "successful_rag_files": len(
                [r for r in rag_results.values() if r["processed"]]
            ),
            "failed_rag_files": len(
                [r for r in rag_results.values() if not r["processed"]]
            ),
        }

    # ==========================================
    # OPTIMIZED BATCH PROCESSING METHODS
    # ==========================================

    async def process_documents_batch_optimized(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        max_concurrent_parsers: int = 4,
        max_concurrent_processors: int = 10,
        enable_progress_tracking: bool = True,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process documents with advanced optimizations for speed

        This method provides significant performance improvements:
        - Concurrent document parsing with prefetching (2-3x faster)
        - Pipeline architecture (parse + process in parallel)
        - Adaptive rate limiting for API calls
        - Progress tracking with ETA estimation
        - Intelligent caching

        Args:
            file_paths: List of file paths to process
            output_dir: Output directory for parsed files
            parse_method: Parsing method to use
            max_concurrent_parsers: Maximum concurrent document parsers (default: 4)
            max_concurrent_processors: Maximum concurrent processors (default: 10)
            enable_progress_tracking: Whether to track and report progress
            progress_callback: Optional callback for progress updates
            **kwargs: Additional processing parameters

        Returns:
            Dict with successful_files, failed_files, and detailed statistics

        Example:
            ```python
            def progress_update(progress):
                print(f"Progress: {progress['percentage']:.1f}%")

            result = await rag.process_documents_batch_optimized(
                file_paths=["doc1.pdf", "doc2.pdf"],
                progress_callback=progress_update
            )
            ```
        """
        # Use config defaults
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method

        self.logger.info(f"Starting optimized batch processing for {len(file_paths)} documents")

        # Create batch optimizer
        optimizer = BatchOptimizer(
            max_concurrent_parsers=max_concurrent_parsers,
            max_concurrent_processors=max_concurrent_processors,
            prefetch_buffer_size=5,
            enable_adaptive_rate=True,
            enable_progress_tracking=enable_progress_tracking,
            logger=self.logger,
        )

        if progress_callback:
            optimizer.set_progress_callback(progress_callback)

        # Process with optimization
        result = await optimizer.process_documents_batch_optimized(
            rag_instance=self,
            file_paths=file_paths,
            output_dir=output_dir,
            parse_method=parse_method,
            **kwargs
        )

        return result

    async def process_folder_optimized(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        max_concurrent_parsers: int = 4,
        max_concurrent_processors: int = 10,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process all files in a folder with optimization

        Args:
            folder_path: Path to folder
            output_dir: Output directory
            parse_method: Parse method
            file_extensions: File extensions to process
            recursive: Process subfolders
            max_concurrent_parsers: Max concurrent parsers
            max_concurrent_processors: Max concurrent processors
            progress_callback: Progress callback function
            **kwargs: Additional parameters

        Returns:
            Processing results and statistics
        """
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if file_extensions is None:
            file_extensions = self.config.supported_file_extensions

        folder_path_obj = Path(folder_path)
        if not folder_path_obj.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Collect files
        files_to_process = []
        for file_ext in file_extensions:
            pattern = f"**/*{file_ext}" if recursive else f"*{file_ext}"
            files_to_process.extend(folder_path_obj.glob(pattern))

        if not files_to_process:
            self.logger.warning(f"No supported files found in {folder_path}")
            return {"successful_files": [], "failed_files": [], "statistics": {}}

        file_paths_str = [str(f) for f in files_to_process]

        return await self.process_documents_batch_optimized(
            file_paths=file_paths_str,
            output_dir=output_dir,
            parse_method=parse_method,
            max_concurrent_parsers=max_concurrent_parsers,
            max_concurrent_processors=max_concurrent_processors,
            progress_callback=progress_callback,
            **kwargs
        )
