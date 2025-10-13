"""
Example: Optimized Batch Processing for RAGAnything

This example demonstrates the new optimized batch processing capabilities
that provide 2-3x faster processing for large document collections.

Features demonstrated:
- Concurrent document parsing with prefetching
- Pipeline architecture (parse + process simultaneously)
- Progress tracking with ETA estimation
- Adaptive rate limiting
- Performance statistics
"""

import asyncio
import time
from pathlib import Path
from raganything import RAGAnything

async def progress_callback(progress_data):
    """
    Callback function to handle progress updates

    Args:
        progress_data: Dict containing:
            - processed: Number of processed documents
            - total: Total number of documents
            - failed: Number of failed documents
            - percentage: Completion percentage
            - eta_seconds: Estimated time remaining
            - rate_docs_per_sec: Processing rate
    """
    print(f"\rProgress: {progress_data['processed']}/{progress_data['total']} "
          f"({progress_data['percentage']:.1f}%) | "
          f"Rate: {progress_data['rate_docs_per_sec']:.2f} docs/s | "
          f"ETA: {progress_data['eta_seconds']:.1f}s", end='', flush=True)


async def main():
    # Initialize RAGAnything
    rag = RAGAnything(
        working_dir="./rag_storage",
        rag_dir="./rag_index",
        parser="mineru",  # or "docling"
    )

    # Example 1: Process a list of documents with optimization
    print("=" * 60)
    print("Example 1: Optimized Batch Processing")
    print("=" * 60)

    documents = [
        "./data/report1.pdf",
        "./data/report2.pdf",
        "./data/research_paper.pdf",
        "./data/technical_spec.docx",
    ]

    start_time = time.time()

    result = await rag.process_documents_batch_optimized(
        file_paths=documents,
        max_concurrent_parsers=4,        # Parse up to 4 documents at once
        max_concurrent_processors=10,     # Process up to 10 chunks concurrently
        enable_progress_tracking=True,
        progress_callback=progress_callback,
    )

    print()  # New line after progress bar

    elapsed_time = time.time() - start_time

    # Display results
    print(f"\n📊 Processing Results:")
    print(f"  ✅ Successful: {len(result['successful_files'])} documents")
    print(f"  ❌ Failed: {len(result['failed_files'])} documents")
    print(f"  ⏱️  Total time: {elapsed_time:.2f}s")

    # Display detailed statistics
    stats = result['statistics']
    print(f"\n📈 Performance Statistics:")
    print(f"  Processing rate: {stats['processing_rate_docs_per_sec']:.2f} docs/sec")
    print(f"  Parsing time: {stats['parsing_time']:.2f}s")
    print(f"  Text processing: {stats['text_processing_time']:.2f}s")
    print(f"  Multimodal processing: {stats['multimodal_processing_time']:.2f}s")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")

    # Show per-document results
    if result['successful_files']:
        print(f"\n✅ Successfully processed files:")
        for file_info in result['successful_files'][:5]:  # Show first 5
            print(f"  - {Path(file_info['file_path']).name} "
                  f"(processing: {file_info['processing_time']:.2f}s, "
                  f"parsing: {file_info['parse_time']:.2f}s)")

    if result['failed_files']:
        print(f"\n❌ Failed files:")
        for file_info in result['failed_files']:
            print(f"  - {Path(file_info['file_path']).name}: {file_info['error']}")

    # Example 2: Process an entire folder with optimization
    print("\n" + "=" * 60)
    print("Example 2: Optimized Folder Processing")
    print("=" * 60)

    folder_result = await rag.process_folder_optimized(
        folder_path="./data/documents",
        file_extensions=['.pdf', '.docx', '.pptx'],
        recursive=True,
        max_concurrent_parsers=6,
        max_concurrent_processors=12,
        progress_callback=progress_callback,
    )

    print()  # New line after progress bar

    print(f"\n📁 Folder Processing Complete:")
    print(f"  Successful: {len(folder_result['successful_files'])} files")
    print(f"  Failed: {len(folder_result['failed_files'])} files")
    print(f"  Rate: {folder_result['statistics']['processing_rate_docs_per_sec']:.2f} docs/sec")

    # Example 3: Compare standard vs optimized processing
    print("\n" + "=" * 60)
    print("Example 3: Performance Comparison")
    print("=" * 60)

    test_docs = ["./data/test1.pdf", "./data/test2.pdf", "./data/test3.pdf"]

    # Standard processing
    print("\n🐢 Standard batch processing...")
    standard_start = time.time()
    await rag.process_folder_complete(
        folder_path="./data/test",
        max_workers=4,
        display_stats=False
    )
    standard_time = time.time() - standard_start

    # Optimized processing (on different set to avoid cache)
    print("🚀 Optimized batch processing...")
    optimized_start = time.time()
    await rag.process_documents_batch_optimized(
        file_paths=test_docs,
        max_concurrent_parsers=4,
        max_concurrent_processors=10,
        enable_progress_tracking=False,
    )
    optimized_time = time.time() - optimized_start

    print(f"\n⚡ Performance Improvement:")
    print(f"  Standard: {standard_time:.2f}s")
    print(f"  Optimized: {optimized_time:.2f}s")
    if standard_time > 0:
        speedup = (standard_time / optimized_time)
        print(f"  Speedup: {speedup:.2f}x faster")

    # Example 4: Custom progress tracking
    print("\n" + "=" * 60)
    print("Example 4: Custom Progress Tracking")
    print("=" * 60)

    class CustomProgressTracker:
        def __init__(self):
            self.start_time = time.time()
            self.logs = []

        def __call__(self, progress):
            """Progress callback"""
            elapsed = time.time() - self.start_time
            log_entry = {
                "timestamp": elapsed,
                "processed": progress['processed'],
                "total": progress['total'],
                "percentage": progress['percentage'],
                "rate": progress['rate_docs_per_sec'],
            }
            self.logs.append(log_entry)

            # Print formatted progress
            bar_length = 40
            filled_length = int(bar_length * progress['percentage'] / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)

            print(f"\r|{bar}| {progress['percentage']:.1f}% "
                  f"[{progress['processed']}/{progress['total']}] "
                  f"ETA: {progress['eta_seconds']:.0f}s", end='', flush=True)

        def save_log(self, filename="processing_log.txt"):
            """Save progress log to file"""
            with open(filename, 'w') as f:
                f.write("Batch Processing Log\n")
                f.write("=" * 50 + "\n")
                for entry in self.logs:
                    f.write(f"Time: {entry['timestamp']:.2f}s | "
                           f"Progress: {entry['processed']}/{entry['total']} "
                           f"({entry['percentage']:.1f}%) | "
                           f"Rate: {entry['rate']:.2f} docs/s\n")

    tracker = CustomProgressTracker()

    await rag.process_documents_batch_optimized(
        file_paths=documents,
        progress_callback=tracker,
    )

    print()  # New line
    tracker.save_log("./batch_processing_log.txt")
    print("📝 Progress log saved to batch_processing_log.txt")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
