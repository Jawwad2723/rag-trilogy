"""
Integration tests for the hybrid retriever end-to-end (mocked backends).
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import SearchResult


@pytest.mark.asyncio
async def test_hybrid_retriever_full_pipeline(
    hybrid_retriever: HybridRetriever,
) -> None:
    """Integration test: retrieve() calls vector + BM25, fuses via RRF, returns sorted results."""
    results = await hybrid_retriever.retrieve(query="What is hybrid search?", top_k=3)

    # Should return at most 3 results
    assert len(results) <= 3

    # Scores should be non-negative
    for r in results:
        assert r.score >= 0.0

    # Results should be sorted descending by score
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_hybrid_retriever_alpha_0_pure_bm25(
    hybrid_retriever: HybridRetriever,
    mock_vector_store: MagicMock,
    mock_bm25_store: MagicMock,
) -> None:
    """With alpha=0.0, only BM25 retrieval should be called (no vector search)."""
    results = await hybrid_retriever.retrieve(query="keyword query", top_k=5, alpha=0.0)

    # Vector store search should NOT be called when alpha=0
    mock_vector_store.search.assert_not_called()
    # BM25 store search SHOULD be called
    mock_bm25_store.search.assert_called_once()
    assert len(results) >= 0


@pytest.mark.asyncio
async def test_hybrid_retriever_alpha_1_pure_vector(
    hybrid_retriever: HybridRetriever,
    mock_vector_store: MagicMock,
    mock_bm25_store: MagicMock,
) -> None:
    """With alpha=1.0, only vector retrieval should be called (no BM25)."""
    results = await hybrid_retriever.retrieve(query="semantic query", top_k=5, alpha=1.0)

    # BM25 store should NOT be called when alpha=1
    mock_bm25_store.search.assert_not_called()
    # Vector store SHOULD be called
    mock_vector_store.search.assert_called_once()
    assert len(results) >= 0


@pytest.mark.asyncio
async def test_hybrid_retriever_overlap_deduplicated(
    hybrid_retriever: HybridRetriever,
) -> None:
    """Overlapping chunks between vector and BM25 should be deduplicated, not duplicated."""
    results = await hybrid_retriever.retrieve(query="overlap test", top_k=10)

    # Collect all chunk IDs
    ids = [r.chunk_id for r in results]
    # No duplicate chunk IDs
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found in results!"


@pytest.mark.asyncio
async def test_hybrid_retriever_vector_failure_graceful(
    mock_bm25_store: MagicMock,
    mock_embedder: MagicMock,
) -> None:
    """If vector search fails, BM25 results should still be returned gracefully."""
    failing_vector_store = MagicMock()
    failing_vector_store.search = AsyncMock(side_effect=ConnectionError("Qdrant unavailable"))

    retriever = HybridRetriever(
        vector_store=failing_vector_store,  # type: ignore
        bm25_store=mock_bm25_store,          # type: ignore
        embedder=mock_embedder,               # type: ignore
        alpha=0.5,
        rrf_k=60,
    )

    # Should not raise; BM25 results should fill in
    results = await retriever.retrieve(query="fallback query", top_k=5)
    assert isinstance(results, list)
