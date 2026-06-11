import os
import time
from typing import List, Dict, Any
from tqdm import tqdm
from src.ingestion.loader import DocumentLoader, calculate_sha256
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import BaseEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.logger import get_logger

logger = get_logger(__name__)

class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    async def ingest_files(self, filepaths: List[str]) -> Dict[str, Any]:
        """Ingests a list of files, skipping already ingested files based on SHA-256 hash."""
        start_time = time.time()
        files_processed = 0
        chunks_created = 0
        chunks_skipped = 0

        # We will progress with tqdm
        for filepath in tqdm(filepaths, desc="Ingesting files"):
            if not os.path.exists(filepath):
                logger.warning("file_skipped_not_found", filepath=filepath)
                continue

            doc_hash = calculate_sha256(filepath)
            
            # Check if this document hash is already ingested in both stores
            # (Strict deduplication - if it exists in Qdrant, we consider it skipped)
            vector_exists = await self.vector_store.document_exists(doc_hash)
            if vector_exists:
                logger.info("file_skipped_already_ingested", filepath=filepath, hash=doc_hash)
                chunks_skipped += 1
                continue

            try:
                # Load pages/documents
                docs = await self.loader.load_file(filepath)
                if not docs:
                    continue

                all_chunks = []
                for doc in docs:
                    # Chunk text
                    chunks = self.chunker.chunk_document(doc.text, doc.metadata)
                    all_chunks.extend(chunks)

                if not all_chunks:
                    continue

                # Generate embeddings for all chunks in a batch
                chunk_texts = [c.text for c in all_chunks]
                embeddings = await self.embedder.embed(chunk_texts)

                # Upsert into vector store and BM25 index
                await self.vector_store.upsert_chunks(all_chunks, embeddings)
                await self.bm25_store.upsert_chunks(all_chunks)

                files_processed += 1
                chunks_created += len(all_chunks)
                logger.info("file_ingestion_success", filepath=filepath, chunks=len(all_chunks))
            except Exception as e:
                logger.error("file_ingestion_failed", filepath=filepath, error=str(e))
                raise e

        # Ensure BM25 store persists its state after new additions
        await self.bm25_store.persist()

        duration = time.time() - start_time
        stats = {
            "files_processed": files_processed,
            "chunks_created": chunks_created,
            "chunks_skipped": chunks_skipped,
            "duration_seconds": round(duration, 2),
        }
        logger.info("ingestion_pipeline_completed", **stats)
        return stats
