from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.config import settings
from src.ingestion.chunker import Chunk
from src.logger import get_logger

logger = get_logger(__name__)

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float

class VectorStore:
    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None
    ) -> None:
        self.url = url or settings.retrieval.vector_store.url or "http://localhost:6333"
        self.collection_name = collection_name or settings.retrieval.vector_store.collection_name or "documents"
        self.vector_size = vector_size or settings.retrieval.embedding.dimensions or 1536
        
        # Async Qdrant Client
        self.client: Optional[AsyncQdrantClient] = None

    async def _get_client(self) -> AsyncQdrantClient:
        if self.client is None:
            # Handle potential localhost replacement if set to empty string
            url_to_use = self.url if self.url else "http://localhost:6333"
            logger.info("connecting_qdrant_client", url=url_to_use)
            self.client = AsyncQdrantClient(url=url_to_use)
            await self._ensure_collection()
        return self.client

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def _ensure_collection(self) -> None:
        assert self.client is not None
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            logger.info("creating_qdrant_collection", collection=self.collection_name, size=self.vector_size)
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                hnsw_config=models.HnswConfigDiff(on_disk=True)
            )

    async def document_exists(self, doc_hash: str) -> bool:
        """Checks if a document with doc_hash already exists in the collection."""
        client = await self._get_client()
        try:
            scroll_result = await client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_hash",
                            match=models.MatchValue(value=doc_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=False
            )
            return len(scroll_result[0]) > 0
        except Exception as e:
            logger.error("qdrant_exists_check_failed", doc_hash=doc_hash, error=str(e))
            return False

    async def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Upserts list of Chunk objects alongside their vector embeddings into Qdrant."""
        client = await self._get_client()
        if not chunks:
            return

        points = []
        for idx, chunk in enumerate(chunks):
            # Combine chunk details and metadata to store in payload
            payload = chunk.metadata.copy()
            payload["text"] = chunk.text
            payload["chunk_id"] = chunk.chunk_id

            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=embeddings[idx],
                    payload=payload
                )
            )

        try:
            await client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info("qdrant_upsert_success", count=len(chunks))
        except Exception as e:
            logger.error("qdrant_upsert_failed", error=str(e))
            raise e

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search the collection using a query vector and return SearchResults."""
        client = await self._get_client()
        
        qdrant_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v)
                    )
                )
            qdrant_filter = models.Filter(must=conditions)

        try:
            results = await client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter
            )

            search_results = []
            for res in results:
                payload = res.payload or {}
                text = payload.get("text", "")
                chunk_id = payload.get("chunk_id", str(res.id))
                
                # Filter out the 'text' field from payload so it isn't duplicated in metadata
                meta = {k: v for k, v in payload.items() if k != "text"}

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=meta,
                        score=res.score
                    )
                )
            return search_results
        except Exception as e:
            logger.error("qdrant_search_failed", error=str(e))
            raise e
