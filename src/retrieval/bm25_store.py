import asyncio
import os
import pickle
import re
from typing import List, Dict, Any, Optional, Set
from rank_bm25 import BM25Okapi
from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import SearchResult
from src.logger import get_logger

logger = get_logger(__name__)

# Basic list of English stopwords to avoid external downloads (nltk/spacy)
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

def preprocess_text(text: str) -> List[str]:
    """Tokenize, lowercase, and remove stopwords from text."""
    # Simple regex word tokenizer
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in STOPWORDS]

class BM25Store:
    def __init__(self, index_path: str = "data/bm25_index.pkl") -> None:
        self.index_path = index_path
        self.chunks: Dict[str, Chunk] = {}  # chunk_id -> Chunk mapping
        self.doc_hashes: Set[str] = set()    # Set of ingested document hashes
        self.bm25: Optional[BM25Okapi] = None
        self._lock = asyncio.Lock()
        
        # Load index if it exists
        self._load_index()

    def _load_index(self) -> None:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", {})
                    self.doc_hashes = data.get("doc_hashes", set())
                
                # Rebuild BM25 instance from persisted corpus
                if self.chunks:
                    corpus = [preprocess_text(c.text) for c in self.chunks.values()]
                    self.bm25 = BM25Okapi(corpus)
                logger.info("bm25_index_loaded", path=self.index_path, chunks=len(self.chunks))
            except Exception as e:
                logger.error("bm25_index_load_failed", path=self.index_path, error=str(e))
                # Start fresh on error
                self.chunks = {}
                self.doc_hashes = set()
                self.bm25 = None

    async def document_exists(self, doc_hash: str) -> bool:
        """Check if doc_hash has already been ingested into BM25."""
        async with self._lock:
            return doc_hash in self.doc_hashes

    async def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """Adds new chunks to the BM25 store and rebuilds the BM25 index."""
        if not chunks:
            return

        async with self._lock:
            loop = asyncio.get_running_loop()
            
            # Update internal records in executor if needed, but it's fast
            for chunk in chunks:
                self.chunks[chunk.chunk_id] = chunk
                doc_hash = chunk.metadata.get("doc_hash")
                if doc_hash:
                    self.doc_hashes.add(doc_hash)

            # Rebuild the BM25Okapi instance
            # Doing tokenization/fitting inside thread pool to avoid blocking event loop
            def rebuild() -> BM25Okapi:
                corpus = [preprocess_text(c.text) for c in self.chunks.values()]
                return BM25Okapi(corpus)

            self.bm25 = await loop.run_in_executor(None, rebuild)
            logger.info("bm25_index_rebuilt", total_chunks=len(self.chunks))

    async def persist(self) -> None:
        """Persists index state to disk asynchronously."""
        async with self._lock:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            loop = asyncio.get_running_loop()

            def save() -> None:
                with open(self.index_path, "wb") as f:
                    pickle.dump({
                        "chunks": self.chunks,
                        "doc_hashes": self.doc_hashes
                    }, f)

            await loop.run_in_executor(None, save)
            logger.info("bm25_index_persisted", path=self.index_path)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Runs BM25 query keyword matching on the corpus and returns SearchResults."""
        async with self._lock:
            if not self.bm25 or not self.chunks:
                return []

            loop = asyncio.get_running_loop()

            # Preprocess query
            tokenized_query = preprocess_text(query)
            
            # Since BM25 scores can be intensive to compute on large datasets, run in executor
            def calculate_scores() -> List[float]:
                assert self.bm25 is not None
                return self.bm25.get_scores(tokenized_query).tolist() # type: ignore

            scores = await loop.run_in_executor(None, calculate_scores)

            # Map scores to chunks
            chunk_list = list(self.chunks.values())
            scored_results = []

            for idx, chunk in enumerate(chunk_list):
                # Apply filters if provided
                if filters:
                    skip = False
                    for fk, fv in filters.items():
                        if chunk.metadata.get(fk) != fv:
                            skip = True
                            break
                    if skip:
                        continue

                score = scores[idx]
                if score > 0:  # Only return chunks with positive matching scores
                    scored_results.append((chunk, score))

            # Sort by score descending
            scored_results.sort(key=lambda x: x[1], reverse=True)
            top_results = scored_results[:top_k]

            search_results = []
            for chunk, score in top_results:
                search_results.append(
                    SearchResult(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        score=float(score)
                    )
                )

            return search_results
