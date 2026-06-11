"""
Shared pytest fixtures and configuration.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.retrieval.vector_store import SearchResult
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.monitoring.cost_tracker import CostTracker


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    return [
        SearchResult(chunk_id="chunk_a1b2", text="The RAG system uses hybrid search.", metadata={"filename": "docs.pdf", "page_number": 1}, score=0.9),
        SearchResult(chunk_id="chunk_c3d4", text="BM25 retrieval provides keyword matching.", metadata={"filename": "docs.pdf", "page_number": 2}, score=0.8),
        SearchResult(chunk_id="chunk_e5f6", text="Ollama runs local LLMs.", metadata={"filename": "guide.md", "page_number": 1}, score=0.7),
    ]


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.search = AsyncMock(return_value=[
        SearchResult(chunk_id="vec_001", text="Vector result 1", metadata={}, score=0.95),
        SearchResult(chunk_id="vec_002", text="Vector result 2", metadata={}, score=0.85),
    ])
    return store


@pytest.fixture
def mock_bm25_store() -> MagicMock:
    store = MagicMock()
    store.search = AsyncMock(return_value=[
        SearchResult(chunk_id="bm25_001", text="BM25 result 1", metadata={}, score=12.5),
        SearchResult(chunk_id="vec_001", text="Overlap result", metadata={}, score=10.0),
    ])
    return store


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1] * 1536])
    return embedder


@pytest.fixture
def hybrid_retriever(mock_vector_store: MagicMock, mock_bm25_store: MagicMock, mock_embedder: MagicMock) -> HybridRetriever:
    return HybridRetriever(
        vector_store=mock_vector_store,  # type: ignore
        bm25_store=mock_bm25_store,      # type: ignore
        embedder=mock_embedder,           # type: ignore
        alpha=0.5,
        rrf_k=60,
    )


@pytest.fixture
def cost_tracker() -> CostTracker:
    return CostTracker(cost_matrix={
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "ollama/*": {"prompt": 0.0, "completion": 0.0},
    })
