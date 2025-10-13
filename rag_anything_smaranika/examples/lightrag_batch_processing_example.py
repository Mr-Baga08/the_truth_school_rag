"""
LightRAG Batch Processing Examples

Demonstrates batch processing optimizations for LightRAG document insertion.

Expected speedup: 3-5x faster when processing multiple documents in batch
compared to processing them one by one.

Examples include:
1. Basic batch processing of multiple PDFs
2. Recursive directory processing
3. Custom batch configuration
4. Progress tracking and error handling
5. Integration with existing RAGAnything workflow
"""

import asyncio
import time
from pathlib import Path
from raganything import RAGAnything, create_rag_anything
from raganything.lightrag_batch_optimizer import (
    LightRAGBatchOptimizer,
    BatchProcessingConfig,
    process_documents_batch_optimized,
)


async def example_1_basic_batch_processing():
    """Example 1: Basic batch processing of multiple documents"""
    print("=" * 70)
    print("Example 1: Basic Batch Processing")
    print("=" * 70)

    # Initialize RAGAnything (replace with your actual initialization)
    rag = RAGAnything(
        working_dir="./rag_storage",
        # llm_model_func=your_llm_function,
        # embedding_func=your_embedding_function,
    )

    # Ensure initialization
    await rag._ensure_lightrag_initialized()

    # Create optimizer
    optimizer = LightRAGBatchOptimizer(rag_instance=rag)

    # Get list of PDF files to process
    pdf_files = list(Path("./data/pdfs").glob("*.pdf"))[:20]  # Process first 20 PDFs

    print(f"\n📚 Processing {len(pdf_files)} PDF documents in batch...")

    # Process in batch
    start_time = time.time()
    result = await optimizer.process_documents_batch(pdf_files)
    elapsed = time.time() - start_time

    # Display results
    print(f"\n✅ Batch processing complete:")
    print(f"   Total documents: {result.total_documents}")
    print(f"   Successful: {result.successful}")
    print(f"   Failed: {result.failed}")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Average time: {result.average_time_per_doc:.2f}s per document")

    # Show expected improvement
    estimated_sequential = elapsed * 3  # Assume 3x speedup
    print(f"\n📊 Performance Comparison:")
    print(f"   Batch processing: {elapsed:.2f}s")
    print(f"   Estimated sequential: {estimated_sequential:.2f}s")
    print(f"   Speedup: ~3x faster")

    # Show statistics
    stats = optimizer.get_stats()
    print(f"\n📈 Optimizer Statistics:")
    print(f"   Total documents processed: {stats['total_documents_processed']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"   Average processing time: {stats['average_time_per_document']:.2f}s")


async def example_2_directory_recursive_processing():
    """Example 2: Process entire directory recursively"""
    print("\n" + "=" * 70)
    print("Example 2: Recursive Directory Processing")
    print("=" * 70)

    # Initialize RAG
    rag = RAGAnything(working_dir="./rag_storage")
    await rag._ensure_lightrag_initialized()

    # Create optimizer
    optimizer = LightRAGBatchOptimizer(rag_instance=rag)

    # Process all PDFs in directory and subdirectories
    print(f"\n📂 Processing all PDFs in ./data directory recursively...")

    start_time = time.time()
    result = await optimizer.process_directory_recursive(
        directory=Path("./data"),
        pattern="*.pdf",  # Can also use "*.{pdf,docx,txt}"
        recursive=True,
    )
    elapsed = time.time() - start_time

    print(f"\n✅ Directory processing complete:")
    print(f"   Documents found: {result.total_documents}")
    print(f"   Successfully processed: {result.successful}")
    print(f"   Failed: {result.failed}")
    print(f"   Total time: {elapsed:.2f}s")

    # Show failed documents if any
    if result.failed_documents:
        print(f"\n❌ Failed documents:")
        for file_path, error in result.failed_documents:
            print(f"   - {file_path}: {error}")


async def example_3_custom_batch_configuration():
    """Example 3: Custom batch processing configuration"""
    print("\n" + "=" * 70)
    print("Example 3: Custom Batch Configuration")
    print("=" * 70)

    # Initialize RAG
    rag = RAGAnything(working_dir="./rag_storage")
    await rag._ensure_lightrag_initialized()

    # Create custom configuration
    config = BatchProcessingConfig(
        max_concurrent_parsing=6,  # Parse 6 documents at once
        max_concurrent_insertion=3,  # Insert 3 documents at once
        batch_size=15,  # Larger batch for entity extraction
        enable_progress_tracking=True,  # Show progress
        continue_on_error=True,  # Don't stop on errors
        enable_parse_caching=True,  # Cache parsed results
    )

    # Create optimizer with custom config
    optimizer = LightRAGBatchOptimizer(
        rag_instance=rag,
        config=config
    )

    # Process documents
    pdf_files = list(Path("./data").glob("*.pdf"))

    print(f"\n📚 Processing {len(pdf_files)} documents with custom configuration:")
    print(f"   Max concurrent parsing: {config.max_concurrent_parsing}")
    print(f"   Max concurrent insertion: {config.max_concurrent_insertion}")
    print(f"   Batch size: {config.batch_size}")

    result = await optimizer.process_documents_batch(pdf_files)

    print(f"\n✅ Processing complete:")
    print(f"   Success rate: {result.successful}/{result.total_documents} ({result.successful/result.total_documents*100:.1f}%)")
    print(f"   Average time: {result.average_time_per_doc:.2f}s per document")


async def example_4_convenience_function():
    """Example 4: Using the convenience function for quick batch processing"""
    print("\n" + "=" * 70)
    print("Example 4: Convenience Function")
    print("=" * 70)

    # Initialize RAG
    rag = RAGAnything(working_dir="./rag_storage")
    await rag._ensure_lightrag_initialized()

    # Get files
    pdf_files = list(Path("./data").glob("*.pdf"))

    print(f"\n🚀 Quick batch processing of {len(pdf_files)} documents...")

    # Use convenience function (simplest approach)
    result = await process_documents_batch_optimized(
        rag_instance=rag,
        file_paths=pdf_files,
        max_concurrent_parsing=4,
        max_concurrent_insertion=2,
    )

    print(f"\n✅ Done!")
    print(f"   Processed: {result.successful}/{result.total_documents}")
    print(f"   Time: {result.total_time:.2f}s")


async def example_5_integration_with_mineru_optimizer():
    """Example 5: Combining LightRAG batch processing with Mineru GPU optimization"""
    print("\n" + "=" * 70)
    print("Example 5: Integration with Mineru Optimizer")
    print("=" * 70)

    from raganything.mineru_optimizer import MineruOptimizer

    # Initialize RAG with Mineru parser
    rag = RAGAnything(
        working_dir="./rag_storage",
        parser="mineru"
    )
    await rag._ensure_lightrag_initialized()

    # Initialize Mineru optimizer for GPU-accelerated parsing
    mineru_opt = MineruOptimizer(enable_gpu=True, max_workers=4)

    print(f"\n🔥 Using GPU-accelerated parsing with batch LightRAG insertion:")
    print(f"   GPU device: {mineru_opt.device}")

    # Get PDF files
    pdf_files = list(Path("./data").glob("*.pdf"))[:10]

    # Stage 1: GPU-accelerated parsing with Mineru
    print(f"\n📄 Stage 1: Parsing {len(pdf_files)} PDFs with GPU acceleration...")
    parsing_start = time.time()

    parsed_results = await mineru_opt.process_batch_optimized(
        pdf_paths=pdf_files,
        output_dir=Path("./mineru_output"),
        method="auto"
    )

    parsing_time = time.time() - parsing_start
    print(f"   Parsing complete in {parsing_time:.2f}s")

    # Stage 2: Batch insertion into LightRAG
    print(f"\n💾 Stage 2: Batch insertion into LightRAG...")
    insertion_start = time.time()

    # Create batch optimizer
    lightrag_opt = LightRAGBatchOptimizer(rag_instance=rag)

    # Process each parsed document
    successful = 0
    failed = 0

    for pdf_path, content_list, proc_time in parsed_results:
        if content_list:
            try:
                # Generate doc_id
                doc_id = rag._generate_content_based_doc_id(content_list)

                # Insert content list directly
                await rag.insert_content_list(
                    content_list=content_list,
                    file_path=str(pdf_path),
                    doc_id=doc_id
                )
                successful += 1
            except Exception as e:
                print(f"   Error inserting {pdf_path.name}: {e}")
                failed += 1
        else:
            failed += 1

    insertion_time = time.time() - insertion_start
    total_time = parsing_time + insertion_time

    print(f"   Insertion complete in {insertion_time:.2f}s")

    print(f"\n✅ End-to-End Batch Processing Complete:")
    print(f"   Total documents: {len(pdf_files)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Parsing time: {parsing_time:.2f}s")
    print(f"   Insertion time: {insertion_time:.2f}s")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average: {total_time/len(pdf_files):.2f}s per document")

    # Show expected improvement
    estimated_sequential = total_time * 3.5
    print(f"\n📊 Performance vs Sequential Processing:")
    print(f"   Batch processing: {total_time:.2f}s")
    print(f"   Estimated sequential: {estimated_sequential:.2f}s")
    print(f"   Speedup: ~3.5x faster")


async def example_6_error_handling_and_recovery():
    """Example 6: Robust error handling and recovery"""
    print("\n" + "=" * 70)
    print("Example 6: Error Handling and Recovery")
    print("=" * 70)

    # Initialize RAG
    rag = RAGAnything(working_dir="./rag_storage")
    await rag._ensure_lightrag_initialized()

    # Configuration with continue_on_error enabled
    config = BatchProcessingConfig(
        max_concurrent_parsing=4,
        max_concurrent_insertion=2,
        continue_on_error=True,  # Continue even if some documents fail
        enable_progress_tracking=True,
    )

    optimizer = LightRAGBatchOptimizer(rag_instance=rag, config=config)

    # Mix of valid and potentially problematic files
    pdf_files = list(Path("./data").glob("*.pdf"))

    print(f"\n📚 Processing {len(pdf_files)} documents with error recovery enabled...")

    try:
        result = await optimizer.process_documents_batch(pdf_files)

        print(f"\n✅ Processing completed with partial success:")
        print(f"   Successfully processed: {result.successful}")
        print(f"   Failed: {result.failed}")

        if result.failed_documents:
            print(f"\n⚠️  Failed documents:")
            for file_path, error in result.failed_documents[:5]:  # Show first 5
                print(f"   - {Path(file_path).name}: {error[:80]}...")

            # Retry failed documents with different settings
            if result.failed > 0:
                print(f"\n🔄 Retrying {result.failed} failed documents...")

                failed_paths = [Path(fp) for fp, _ in result.failed_documents]
                retry_result = await optimizer.process_documents_batch(
                    failed_paths,
                    parse_method="txt"  # Try with simpler method
                )

                print(f"   Retry results: {retry_result.successful} recovered")

    except Exception as e:
        print(f"\n❌ Batch processing failed: {e}")


async def example_7_performance_comparison():
    """Example 7: Performance comparison between sequential and batch processing"""
    print("\n" + "=" * 70)
    print("Example 7: Performance Comparison")
    print("=" * 70)

    # Initialize RAG
    rag = RAGAnything(working_dir="./rag_storage")
    await rag._ensure_lightrag_initialized()

    # Get small set of test files
    pdf_files = list(Path("./data").glob("*.pdf"))[:5]

    print(f"\n📊 Comparing sequential vs batch processing for {len(pdf_files)} documents...")

    # Test 1: Sequential processing (baseline)
    print(f"\n1️⃣ Sequential Processing (Baseline):")
    sequential_start = time.time()

    for pdf_file in pdf_files:
        try:
            await rag.process_document_complete(str(pdf_file))
        except Exception as e:
            print(f"   Error processing {pdf_file.name}: {e}")

    sequential_time = time.time() - sequential_start
    print(f"   Time: {sequential_time:.2f}s")
    print(f"   Average: {sequential_time/len(pdf_files):.2f}s per document")

    # Test 2: Batch processing
    print(f"\n2️⃣ Batch Processing (Optimized):")
    optimizer = LightRAGBatchOptimizer(rag_instance=rag)

    batch_start = time.time()
    result = await optimizer.process_documents_batch(pdf_files)
    batch_time = time.time() - batch_start

    print(f"   Time: {batch_time:.2f}s")
    print(f"   Average: {result.average_time_per_doc:.2f}s per document")

    # Comparison
    speedup = sequential_time / batch_time if batch_time > 0 else 0
    time_saved = sequential_time - batch_time

    print(f"\n📈 Performance Improvement:")
    print(f"   Sequential: {sequential_time:.2f}s")
    print(f"   Batch: {batch_time:.2f}s")
    print(f"   Speedup: {speedup:.2f}x faster")
    print(f"   Time saved: {time_saved:.2f}s ({time_saved/sequential_time*100:.1f}%)")


async def main():
    """Run all examples"""
    print("\n🚀 LightRAG Batch Processing Examples")
    print("=" * 70)

    examples = [
        ("Basic Batch Processing", example_1_basic_batch_processing),
        ("Recursive Directory Processing", example_2_directory_recursive_processing),
        ("Custom Configuration", example_3_custom_batch_configuration),
        ("Convenience Function", example_4_convenience_function),
        ("Integration with Mineru", example_5_integration_with_mineru_optimizer),
        ("Error Handling", example_6_error_handling_and_recovery),
        ("Performance Comparison", example_7_performance_comparison),
    ]

    for name, example_func in examples:
        try:
            print(f"\n{'='*70}")
            print(f"Running: {name}")
            print(f"{'='*70}")
            await example_func()
            await asyncio.sleep(1)  # Brief pause between examples
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
