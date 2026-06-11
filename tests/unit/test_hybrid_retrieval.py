"""
Unit tests for RRF fusion algorithm.
"""
import pytest
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import SearchResult


def make_retriever() -> HybridRetriever:
    return HybridRetriever(
        vector_store=None,  # type: ignore
        bm25_store=None,    # type: ignore
        embedder=None,      # type: ignore
        rrf_k=60,
    )


def test_rrf_overlap_chunk_scored_higher() -> None:
    """A chunk appearing in both vector and BM25 lists should outrank chunks in only one list."""
    retriever = make_retriever()

    vector_results = [
        SearchResult(chunk_id="doc1", text="text1", metadata={}, score=0.9),
        SearchResult(chunk_id="doc2", text="text2", metadata={}, score=0.8),
    ]
    bm25_results = [
        SearchResult(chunk_id="doc2", text="text2", metadata={}, score=12.5),
        SearchResult(chunk_id="doc3", text="text3", metadata={}, score=10.0),
    ]

    fused = retriever._rrf_fusion(vector_results, bm25_results, alpha=0.5, top_k=3)

    assert len(fused) == 3
    # doc2 appears in both lists and should rank first
    assert fused[0].chunk_id == "doc2"


def test_rrf_pure_vector_alpha_1() -> None:
    """With alpha=1.0, only vector results contribute to scoring."""
    retriever = make_retriever()

    vector_results = [
        SearchResult(chunk_id="vec1", text="v1", metadata={}, score=0.9),
    ]
    bm25_results = [
        SearchResult(chunk_id="bm1", text="b1", metadata={}, score=99.0),
    ]

    fused = retriever._rrf_fusion(vector_results, bm25_results, alpha=1.0, top_k=3)
    ids = [r.chunk_id for r in fused]

    # BM25 result should score 0 (weight = 1-alpha = 0) and not appear (or appear with 0 score)
    # vec1 should be in results
    assert "vec1" in ids
    # bm1 should have 0 score contribution from alpha=1.0
    bm1_result = next((r for r in fused if r.chunk_id == "bm1"), None)
    if bm1_result:
        assert bm1_result.score == 0.0


def test_rrf_no_duplicates() -> None:
    """Chunks appearing in both lists should only appear once in output."""
    retriever = make_retriever()

    overlap = SearchResult(chunk_id="shared", text="shared text", metadata={}, score=0.5)
    vector_results = [overlap]
    bm25_results = [overlap]

    fused = retriever._rrf_fusion(vector_results, bm25_results, alpha=0.5, top_k=5)
    ids = [r.chunk_id for r in fused]
    assert ids.count("shared") == 1


def test_rrf_empty_inputs() -> None:
    """Empty inputs should return an empty list without errors."""
    retriever = make_retriever()
    fused = retriever._rrf_fusion([], [], alpha=0.5, top_k=5)
    assert fused == []


def test_rrf_top_k_respected() -> None:
    """Output should never exceed top_k results."""
    retriever = make_retriever()

    vector_results = [
        SearchResult(chunk_id=f"v{i}", text=f"text {i}", metadata={}, score=float(i))
        for i in range(10)
    ]
    bm25_results = [
        SearchResult(chunk_id=f"b{i}", text=f"text {i}", metadata={}, score=float(i))
        for i in range(10)
    ]

    fused = retriever._rrf_fusion(vector_results, bm25_results, alpha=0.5, top_k=3)
    assert len(fused) == 3


def test_rrf_score_formula() -> None:
    """Verify the RRF score formula is correctly applied."""
    retriever = make_retriever()  # rrf_k=60

    vector_results = [
        SearchResult(chunk_id="doc1", text="text1", metadata={}, score=0.9),  # rank 0
    ]
    bm25_results = [
        SearchResult(chunk_id="doc1", text="text1", metadata={}, score=10.0),  # rank 0
    ]

    fused = retriever._rrf_fusion(vector_results, bm25_results, alpha=0.5, top_k=1)
    assert len(fused) == 1

    # Expected: 0.5/(60+0+1) + 0.5/(60+0+1) = 1.0/61
    expected = 1.0 / 61.0
    assert abs(fused[0].score - expected) < 1e-9
