import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import SearchResult

@pytest.mark.asyncio
async def test_reranker_sorting() -> None:
    reranker = CrossEncoderReranker(enabled=True)
    
    # Mock model prediction scores
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5]
    
    results = [
        SearchResult(chunk_id="c1", text="Text 1", metadata={}, score=0.5),
        SearchResult(chunk_id="c2", text="Text 2", metadata={}, score=0.6),
        SearchResult(chunk_id="c3", text="Text 3", metadata={}, score=0.7)
    ]
    
    with patch.object(reranker, "_get_model", return_value=mock_model):
        reranked = await reranker.rerank("query", results, top_n=2)
        
        # Sorted order by prediction score: c2 (0.9), c3 (0.5), c1 (0.1)
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "c2"
        assert reranked[0].score == 0.9
        assert reranked[1].chunk_id == "c3"
        assert reranked[1].score == 0.5

@pytest.mark.asyncio
async def test_reranker_disabled() -> None:
    reranker = CrossEncoderReranker(enabled=False)
    results = [
        SearchResult(chunk_id="c1", text="Text 1", metadata={}, score=0.5),
        SearchResult(chunk_id="c2", text="Text 2", metadata={}, score=0.6)
    ]
    # Reranking is disabled, should return sliced list
    reranked = await reranker.rerank("query", results, top_n=1)
    assert len(reranked) == 1
    assert reranked[0].chunk_id == "c1"

@pytest.mark.asyncio
async def test_reranker_fallback_on_error() -> None:
    reranker = CrossEncoderReranker(enabled=True)
    
    results = [
        SearchResult(chunk_id="c1", text="Text 1", metadata={}, score=0.5),
        SearchResult(chunk_id="c2", text="Text 2", metadata={}, score=0.6)
    ]
    
    # Cause _get_model to fail
    with patch.object(reranker, "_get_model", side_effect=RuntimeError("GPU out of memory")):
        reranked = await reranker.rerank("query", results, top_n=1)
        
        # Should fallback gracefully to returning first slice
        assert len(reranked) == 1
        assert reranked[0].chunk_id == "c1"
