import pytest
from unittest.mock import MagicMock
from src.generation.chain import RAGChain

def get_mock_chain() -> RAGChain:
    retriever = MagicMock()
    reranker = MagicMock()
    cost_tracker = MagicMock()
    return RAGChain(retriever=retriever, reranker=reranker, cost_tracker=cost_tracker)

def test_no_violation_with_citations() -> None:
    chain = get_mock_chain()
    
    # 3 sentences, with citation hash marker
    text = "The retrieval uses RRF [chunk_abc]. It runs in parallel [chunk_def]. Reranking optimizes it [chunk_xyz]."
    assert not chain._check_citation_violation(text)
    
    # Using source filename citation marker
    text2 = "This is a statement. Another fact [SOURCE: docs.pdf, page 3]. RAG is production grade."
    assert not chain._check_citation_violation(text2)

def test_violation_without_citations() -> None:
    chain = get_mock_chain()
    
    # 3 sentences, 0 citation markers
    text = "The retrieval uses hybrid search. It runs in parallel. Reranking optimizes it."
    assert chain._check_citation_violation(text)

def test_no_violation_short_answer() -> None:
    chain = get_mock_chain()
    
    # Under 2 sentences, citation not strictly enforced
    text = "The system is fully local."
    assert not chain._check_citation_violation(text)

def test_refusal_is_exempt() -> None:
    chain = get_mock_chain()
    
    text = "I don't have enough information in the provided documents to answer this question."
    assert not chain._check_citation_violation(text)
