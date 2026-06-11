from typing import Dict
from src.config import settings
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import BaseEmbedder, OpenAIEmbedder, LocalEmbedder
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.chain import RAGChain
from src.generation.llm_client import BaseLLMClient, OpenAIClient, AnthropicClient, OllamaClient
from src.monitoring.cost_tracker import CostTracker

# Singletons for reuse across API calls
_vector_store: VectorStore | None = None
_bm25_store: BM25Store | None = None
_embedder: BaseEmbedder | None = None
_chunker: Chunker | None = None
_reranker: CrossEncoderReranker | None = None
_cost_tracker: CostTracker | None = None
_rag_chain: RAGChain | None = None
_pipeline: IngestionPipeline | None = None

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            url=settings.retrieval.vector_store.url,
            collection_name=settings.retrieval.vector_store.collection_name,
            vector_size=settings.retrieval.embedding.dimensions
        )
    return _vector_store

def get_bm25_store() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store(index_path="data/bm25_index.pkl")
    return _bm25_store

def get_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        provider = settings.retrieval.embedding.provider
        model = settings.retrieval.embedding.model
        if provider == "openai":
            _embedder = OpenAIEmbedder(model=model)
        else:
            _embedder = LocalEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def get_chunker() -> Chunker:
    global _chunker
    if _chunker is None:
        # Default strategy: semantic. Support fixed/sentence from config
        _chunker = Chunker(
            strategy="semantic",  # type: ignore
            chunk_size=512,
            chunk_overlap=64
        )
    return _chunker

def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model_name=settings.retrieval.reranker.model,
            enabled=settings.retrieval.reranker.enabled
        )
    return _reranker

def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker

def get_document_loader() -> DocumentLoader:
    return DocumentLoader()

def get_ingestion_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(
            loader=get_document_loader(),
            chunker=get_chunker(),
            embedder=get_embedder(),
            vector_store=get_vector_store(),
            bm25_store=get_bm25_store()
        )
    return _pipeline

def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        retriever = HybridRetriever(
            vector_store=get_vector_store(),
            bm25_store=get_bm25_store(),
            embedder=get_embedder(),
            alpha=settings.retrieval.hybrid.alpha,
            rrf_k=settings.retrieval.hybrid.rrf_k
        )
        
        _rag_chain = RAGChain(
            retriever=retriever,
            reranker=get_reranker(),
            cost_tracker=get_cost_tracker()
        )
    return _rag_chain
