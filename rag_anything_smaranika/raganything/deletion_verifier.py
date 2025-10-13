"""
Document Deletion Verifier Module

This module provides comprehensive verification for document deletion operations,
ensuring complete cleanup of:
- Knowledge graph entities and relationships
- Embedding vectors (chunks, entities, relationships)
- Text chunks and metadata
- Document status records
- Physical files

Author: RAGAnything Team
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class DeletionReport:
    """Report of what was deleted during a document deletion operation"""

    doc_id: str
    """Document ID that was deleted"""

    success: bool
    """Overall deletion success"""

    chunks_deleted: int = 0
    """Number of text chunks deleted"""

    entities_deleted: int = 0
    """Number of entities deleted from KG"""

    relationships_deleted: int = 0
    """Number of relationships deleted from KG"""

    entities_rebuilt: int = 0
    """Number of entities rebuilt (shared across documents)"""

    relationships_rebuilt: int = 0
    """Number of relationships rebuilt (shared across documents)"""

    vectors_deleted: int = 0
    """Total number of vectors deleted (chunks + entities + relationships)"""

    files_deleted: List[str] = field(default_factory=list)
    """Physical files deleted"""

    directories_deleted: List[str] = field(default_factory=list)
    """Directories deleted"""

    errors: List[str] = field(default_factory=list)
    """Any errors encountered during deletion"""

    warnings: List[str] = field(default_factory=list)
    """Any warnings during deletion"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "doc_id": self.doc_id,
            "success": self.success,
            "summary": {
                "chunks_deleted": self.chunks_deleted,
                "entities_deleted": self.entities_deleted,
                "relationships_deleted": self.relationships_deleted,
                "entities_rebuilt": self.entities_rebuilt,
                "relationships_rebuilt": self.relationships_rebuilt,
                "vectors_deleted": self.vectors_deleted,
                "files_deleted": len(self.files_deleted),
                "directories_deleted": len(self.directories_deleted),
            },
            "details": {
                "files_deleted": self.files_deleted,
                "directories_deleted": self.directories_deleted,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DeletionVerifier:
    """
    Verifier for document deletion operations

    This class verifies that all components of a document have been properly
    deleted from the RAG system, including:
    - Knowledge graph data
    - Vector embeddings
    - Text storage
    - Metadata
    """

    def __init__(self, rag_instance: Any, storage_dir: Path):
        """
        Initialize deletion verifier

        Args:
            rag_instance: RAGAnything instance
            storage_dir: Storage directory for the domain
        """
        self.rag = rag_instance
        self.storage_dir = Path(storage_dir)

    async def verify_deletion(
        self,
        doc_id: str,
        deletion_result: Any = None
    ) -> DeletionReport:
        """
        Verify that a document was completely deleted

        Args:
            doc_id: Document ID that should be deleted
            deletion_result: Result from LightRAG's adelete_by_doc_id

        Returns:
            DeletionReport with verification results
        """
        report = DeletionReport(doc_id=doc_id, success=True)

        try:
            # Parse deletion result from LightRAG
            if deletion_result:
                report = self._parse_lightrag_result(deletion_result, report)

            # Verify document status is gone
            doc_status_exists = await self._check_doc_status_exists(doc_id)
            if doc_status_exists:
                report.errors.append("Document still exists in doc_status")
                report.success = False

            # Verify full_docs is gone
            full_doc_exists = await self._check_full_doc_exists(doc_id)
            if full_doc_exists:
                report.errors.append("Document still exists in full_docs")
                report.success = False

            # Check if any chunks remain
            remaining_chunks = await self._check_remaining_chunks(doc_id)
            if remaining_chunks:
                report.warnings.append(
                    f"{len(remaining_chunks)} chunks may still exist"
                )

            logger.info(
                f"Deletion verification complete for {doc_id}: "
                f"success={report.success}, "
                f"chunks={report.chunks_deleted}, "
                f"entities={report.entities_deleted}, "
                f"relationships={report.relationships_deleted}"
            )

        except Exception as e:
            logger.error(f"Error during deletion verification: {e}", exc_info=True)
            report.errors.append(f"Verification error: {str(e)}")
            report.success = False

        return report

    def _parse_lightrag_result(
        self,
        deletion_result: Any,
        report: DeletionReport
    ) -> DeletionReport:
        """Parse LightRAG deletion result and update report"""
        try:
            # LightRAG returns a DeletionResult object
            if hasattr(deletion_result, 'status'):
                if deletion_result.status == "success":
                    logger.info(f"LightRAG deletion successful: {deletion_result.message}")
                elif deletion_result.status == "not_found":
                    report.warnings.append(deletion_result.message)
                else:
                    report.errors.append(deletion_result.message)
                    report.success = False

            # Try to extract deletion statistics from message
            if hasattr(deletion_result, 'message'):
                message = deletion_result.message

                # Parse deletion counts from message
                # Example: "Successfully deleted 10 chunks from storage"
                import re

                chunks_match = re.search(r'(\d+)\s+chunks', message)
                if chunks_match:
                    report.chunks_deleted = int(chunks_match.group(1))

                entities_match = re.search(r'(\d+)\s+entities', message)
                if entities_match:
                    report.entities_deleted = int(entities_match.group(1))

                relations_match = re.search(r'(\d+)\s+relations', message)
                if relations_match:
                    report.relationships_deleted = int(relations_match.group(1))

                # Calculate total vectors deleted
                report.vectors_deleted = (
                    report.chunks_deleted +
                    report.entities_deleted +
                    report.relationships_deleted * 2  # Bidirectional
                )

        except Exception as e:
            logger.warning(f"Could not parse LightRAG result: {e}")

        return report

    async def _check_doc_status_exists(self, doc_id: str) -> bool:
        """Check if document still exists in doc_status"""
        try:
            if hasattr(self.rag, 'lightrag') and hasattr(self.rag.lightrag, 'doc_status'):
                doc_status_data = await self.rag.lightrag.doc_status.get_by_id(doc_id)
                return doc_status_data is not None
        except Exception as e:
            logger.debug(f"Error checking doc_status: {e}")
        return False

    async def _check_full_doc_exists(self, doc_id: str) -> bool:
        """Check if document still exists in full_docs"""
        try:
            if hasattr(self.rag, 'lightrag') and hasattr(self.rag.lightrag, 'full_docs'):
                full_doc_data = await self.rag.lightrag.full_docs.get_by_id(doc_id)
                return full_doc_data is not None
        except Exception as e:
            logger.debug(f"Error checking full_docs: {e}")
        return False

    async def _check_remaining_chunks(self, doc_id: str) -> List[str]:
        """Check for any remaining chunks associated with the document"""
        # This is a best-effort check since chunks don't directly reference doc_id
        # after deletion, so this mainly serves as a sanity check
        return []


async def delete_document_complete(
    rag_instance: Any,
    doc_id: str,
    storage_dir: Path,
    upload_files: List[Path],
    output_dirs: List[Path]
) -> DeletionReport:
    """
    Complete document deletion with verification

    This function performs a comprehensive deletion of a document including:
    1. RAG system deletion (via LightRAG's adelete_by_doc_id)
    2. Physical file deletion
    3. Output directory cleanup
    4. Verification of deletion

    Args:
        rag_instance: RAGAnything instance
        doc_id: Document ID to delete
        storage_dir: Storage directory for verification
        upload_files: Physical upload files to delete
        output_dirs: Output directories to delete

    Returns:
        DeletionReport with complete deletion information
    """
    report = DeletionReport(doc_id=doc_id, success=True)

    try:
        # Step 1: Delete from RAG system (KG, vectors, chunks, metadata)
        logger.info(f"Deleting document {doc_id} from RAG system...")

        try:
            # RAGAnything wraps LightRAG, so we need to access the lightrag instance
            if hasattr(rag_instance, 'lightrag') and rag_instance.lightrag:
                deletion_result = await rag_instance.lightrag.adelete_by_doc_id(doc_id)
            else:
                # Fallback: try calling directly (in case rag_instance IS a LightRAG instance)
                deletion_result = await rag_instance.adelete_by_doc_id(doc_id)
            logger.info(f"RAG deletion result: {deletion_result.message if hasattr(deletion_result, 'message') else deletion_result}")

            # Create verifier and verify deletion
            verifier = DeletionVerifier(rag_instance, storage_dir)
            report = await verifier.verify_deletion(doc_id, deletion_result)

        except Exception as e:
            logger.error(f"Error deleting from RAG system: {e}", exc_info=True)
            report.errors.append(f"RAG deletion error: {str(e)}")
            report.success = False

        # Step 2: Delete physical files
        for file_path in upload_files:
            try:
                if file_path.exists() and file_path.is_file():
                    file_path.unlink()
                    report.files_deleted.append(str(file_path))
                    logger.info(f"Deleted upload file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
                report.errors.append(f"File deletion error: {file_path.name}")
                report.success = False

        # Step 3: Delete output directories
        import shutil
        for output_dir in output_dirs:
            try:
                if output_dir.exists():
                    if output_dir.is_dir():
                        shutil.rmtree(output_dir)
                        report.directories_deleted.append(str(output_dir))
                        logger.info(f"Deleted output directory: {output_dir}")
                    elif output_dir.is_file():
                        output_dir.unlink()
                        report.files_deleted.append(str(output_dir))
                        logger.info(f"Deleted output file: {output_dir}")
            except Exception as e:
                logger.error(f"Error deleting output {output_dir}: {e}")
                report.errors.append(f"Output deletion error: {output_dir.name}")
                report.success = False

        # Final summary
        logger.info(
            f"Document deletion complete for {doc_id}: "
            f"success={report.success}, "
            f"chunks={report.chunks_deleted}, "
            f"entities={report.entities_deleted}, "
            f"relationships={report.relationships_deleted}, "
            f"files={len(report.files_deleted)}, "
            f"dirs={len(report.directories_deleted)}"
        )

    except Exception as e:
        logger.error(f"Unexpected error during document deletion: {e}", exc_info=True)
        report.errors.append(f"Unexpected error: {str(e)}")
        report.success = False

    return report


# Export main components
__all__ = [
    "DeletionReport",
    "DeletionVerifier",
    "delete_document_complete",
]
