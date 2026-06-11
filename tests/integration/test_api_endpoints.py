"""
Integration tests for the FastAPI endpoints using TestClient (no live services needed).
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from src.api.main import app
from src.generation.chain import RAGChain
from src.api.deps import get_rag_chain, get_ingestion_pipeline


@pytest.fixture
def mock_rag_chain() -> MagicMock:
    chain = MagicMock(spec=RAGChain)
    chain.query = AsyncMock(return_value={
        "answer": "The system uses RRF [chunk_abc].",
        "citations": [
            {
                "chunk_id": "chunk_abc",
                "source": "docs.pdf",
                "page": 1,
                "text_preview": "The system uses hybrid search."
            }
        ],
        "metadata": {
            "retrieval_latency_ms": 45.2,
            "reranking_latency_ms": 120.3,
            "generation_latency_ms": 850.0,
            "total_latency_ms": 1015.5,
            "model_used": "gpt-4o-mini",
            "prompt_tokens": 512,
            "completion_tokens": 128,
            "estimated_cost_usd": 0.000154,
            "chunks_retrieved": 5,
            "citation_violations": False,
        }
    })
    return chain


@pytest.fixture
def test_client(mock_rag_chain: MagicMock) -> TestClient:
    """Returns a TestClient with mocked RAG chain dependency."""
    app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain
    client = TestClient(app, raise_server_exceptions=True)
    yield client
    app.dependency_overrides.clear()


def test_health_endpoint(test_client: TestClient) -> None:
    """GET /health should return 200 with status ok."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_metrics_endpoint(test_client: TestClient) -> None:
    """GET /metrics should return 200 with Prometheus text format."""
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "rag_requests_total" in response.text or "# HELP" in response.text


def test_query_endpoint_success(test_client: TestClient, mock_rag_chain: MagicMock) -> None:
    """POST /query should return answer and citations."""
    payload = {
        "question": "What is hybrid search?",
        "top_k": 5,
        "alpha": 0.5,
        "model": "default",
        "stream": False,
    }
    response = test_client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "metadata" in data
    assert data["answer"] == "The system uses RRF [chunk_abc]."
    mock_rag_chain.query.assert_called_once()


def test_query_endpoint_missing_question(test_client: TestClient) -> None:
    """POST /query with missing question should return 422."""
    response = test_client.post("/query", json={"top_k": 5})
    assert response.status_code == 422


def test_query_endpoint_invalid_alpha(test_client: TestClient) -> None:
    """POST /query with valid structure should succeed even if alpha is at boundary."""
    payload = {
        "question": "test question",
        "alpha": 0.0,
        "model": "default",
        "stream": False,
    }
    response = test_client.post("/query", json=payload)
    # Should succeed (alpha=0.0 is valid — BM25 only mode)
    assert response.status_code == 200
