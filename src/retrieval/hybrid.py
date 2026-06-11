import time
from typing import List, Dict, Any, Optional
from src.retrieval.vector_store import VectorStore, SearchResult
from src.retrieval.bm25_store import BM25Store
from src.ingestion.embedder import BaseEmbedder
from src.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: BaseEmbedder,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder
        self.alpha = alpha
        self.rrf_k = rrf_k

    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        alpha: float,
        top_k: int,
    ) -> List[SearchResult]:
        """
        Combines vector and BM25 ranked lists via Reciprocal Rank Fusion (RRF).
        Score = alpha * (1 / (rrf_k + rank_vector+1)) + (1-alpha) * (1 / (rrf_k + rank_bm25+1))
        Ranks are 1-indexed (enumerate starts at 0, +1 applied inside formula).
        """
        fused_scores: Dict[str, float] = {}
        chunks_map: Dict[str, SearchResult] = {}

        for rank, res in enumerate(vector_results):
            chunks_map[res.chunk_id] = res
            fused_scores[res.chunk_id] = fused_scores.get(res.chunk_id, 0.0) + (
                alpha / (self.rrf_k + rank + 1)
            )

        for rank, res in enumerate(bm25_results):
            chunks_map[res.chunk_id] = res
            fused_scores[res.chunk_id] = fused_scores.get(res.chunk_id, 0.0) + (
                (1.0 - alpha) / (self.rrf_k + rank + 1)
            )

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        return [
            SearchResult(
                chunk_id=cid,
                text=chunks_map[cid].text,
                metadata=chunks_map[cid].metadata,
                score=fused_scores[cid],
            )
            for cid in sorted_ids[:top_k]
        ]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Retrieves top-k documents using a hybrid approach (Vector + BM25) combined via RRF.
        alpha: 0.0 = BM25 only, 1.0 = Vector only, 0.5 = Balanced (default)
        """
        active_alpha = alpha if alpha is not None else self.alpha

        vector_results: List[SearchResult] = []
        bm25_results: List[SearchResult] = []
        vec_latency = 0.0
        bm_latency = 0.0

        # Vector Retrieval
        if active_alpha > 0.0:
            start_vec = time.time()
            try:
                query_vector = (await self.embedder.embed([query]))[0]
                vector_results = await self.vector_store.search(
                    query_vector=query_vector,
                    top_k=top_k * 2,
                    filters=filters,
                )
            except Exception as e:
                logger.error("hybrid_vector_retrieval_failed", error=str(e))
            vec_latency = (time.time() - start_vec) * 1000
            logger.info("vector_retrieval_latency", latency_ms=vec_latency, results=len(vector_results))

        # BM25 Retrieval
        if active_alpha < 1.0:
            start_bm = time.time()
            try:
                bm25_results = await self.bm25_store.search(
                    query=query,
                    top_k=top_k * 2,
                    filters=filters,
                )
            except Exception as e:
                logger.error("hybrid_bm25_retrieval_failed", error=str(e))
            bm_latency = (time.time() - start_bm) * 1000
            logger.info("bm25_retrieval_latency", latency_ms=bm_latency, results=len(bm25_results))

        # RRF Fusion
        final_results = self._rrf_fusion(vector_results, bm25_results, active_alpha, top_k)

        logger.info(
            "hybrid_retrieval_completed",
            total_results=len(final_results),
            alpha=active_alpha,
            vector_latency_ms=vec_latency,
            bm25_latency_ms=bm_latency,
        )
        return final_results
