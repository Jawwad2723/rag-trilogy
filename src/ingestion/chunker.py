import hashlib
import re
from typing import List, Literal
from pydantic import BaseModel, Field
from src.logger import get_logger

logger = get_logger(__name__)

class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict = Field(default_factory=dict)

class Chunker:
    def __init__(
        self,
        strategy: Literal["fixed", "sentence", "semantic"] = "fixed",
        chunk_size: int = 512,  # in tokens (approx)
        chunk_overlap: int = 64,  # in tokens (approx)
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using word-based splitting (approx 1.3 words per token)."""
        # Alternatively, character count / 4. Let's combine them:
        # A standard token is ~4 characters or ~0.75 words.
        words = text.split()
        return int(len(words) * 1.3)

    def chunk_document(self, text: str, parent_metadata: dict) -> List[Chunk]:
        """Chunks a document text and enriches it with chunk metadata and chunk ID."""
        doc_hash = parent_metadata.get("doc_hash", "")
        if not doc_hash:
            # Generate one if not provided
            doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if self.strategy == "sentence":
            raw_chunks = self._sentence_chunking(text)
        elif self.strategy == "semantic":
            # Semantic chunking fallback: group sentences by similarity or paragraph breaks
            raw_chunks = self._semantic_chunking(text)
        else:
            # Default to fixed token chunking
            raw_chunks = self._fixed_chunking(text)

        chunks = []
        total_chunks = len(raw_chunks)

        for idx, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            # Generate deterministic chunk_id: SHA-256(doc_hash + str(idx))
            chunk_id_src = f"{doc_hash}_{idx}"
            chunk_id = hashlib.sha256(chunk_id_src.encode("utf-8")).hexdigest()

            # Merge parent metadata with chunk info
            metadata = parent_metadata.copy()
            metadata.update({
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "start_char": start_char,
                "end_char": end_char,
                "chunk_id": chunk_id,
            })

            chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))

        logger.info("document_chunked", strategy=self.strategy, num_chunks=len(chunks))
        return chunks

    def _fixed_chunking(self, text: str) -> List[tuple[str, int, int]]:
        """Chunk by word count approximation (since 1 token is approx 0.75 words)."""
        words = text.split()
        # approximate chunk size in words: size * 0.75
        words_per_chunk = int(self.chunk_size * 0.75)
        overlap_words = int(self.chunk_overlap * 0.75)

        if words_per_chunk <= 0:
            words_per_chunk = 100
        if overlap_words >= words_per_chunk:
            overlap_words = words_per_chunk // 2

        chunks = []
        i = 0
        total_words = len(words)

        if total_words == 0:
            return []

        while i < total_words:
            # Get slice of words
            chunk_words = words[i : i + words_per_chunk]
            chunk_text = " ".join(chunk_words)

            # Find char offsets of this chunk in original text
            # To handle duplicate substrings, we search from last i index
            # Let's approximate start_char / end_char
            # We can find them by searching the first match or tracking the exact offsets
            # A robust search finds the substring index
            # Since words might be joined, let's search from the start of the document
            # But a simple regex search or index search is fine.
            try:
                start_char = text.find(chunk_text)
                if start_char == -1:
                    # If direct search fails due to spacing discrepancies
                    start_char = 0
                    end_char = len(text)
                else:
                    end_char = start_char + len(chunk_text)
            except Exception:
                start_char = 0
                end_char = len(text)

            chunks.append((chunk_text, start_char, end_char))

            i += (words_per_chunk - overlap_words)
            if i >= total_words or len(chunk_words) < words_per_chunk:
                break

        return chunks

    def _sentence_chunking(self, text: str) -> List[tuple[str, int, int]]:
        """Split text into sentences and group them to fit inside chunk_size."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        
        start_char = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            tokens = self.estimate_tokens(sentence)
            if current_tokens + tokens > self.chunk_size and current_chunk_sentences:
                # Flush current chunk
                chunk_text = " ".join(current_chunk_sentences)
                idx = text.find(chunk_text, start_char)
                if idx != -1:
                    start_char = idx
                end_char = start_char + len(chunk_text)
                chunks.append((chunk_text, start_char, end_char))
                
                # Setup next chunk with overlap (keep last sentence if overlap is configured)
                # Keep last 1 or 2 sentences depending on overlap size
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk_sentences):
                    s_tokens = self.estimate_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + tokens
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += tokens

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            idx = text.find(chunk_text, start_char)
            if idx != -1:
                start_char = idx
            end_char = start_char + len(chunk_text)
            chunks.append((chunk_text, start_char, end_char))

        return chunks

    def _semantic_chunking(self, text: str) -> List[tuple[str, int, int]]:
        """
        Group paragraphs together. If paragraphs are too large, 
        split them via sentence chunking.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk_paras = []
        current_tokens = 0
        start_char = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.estimate_tokens(para)
            if para_tokens > self.chunk_size:
                # If a paragraph is larger than chunk_size, split it using sentence chunking
                if current_chunk_paras:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    idx = text.find(chunk_text, start_char)
                    if idx != -1:
                        start_char = idx
                    end_char = start_char + len(chunk_text)
                    chunks.append((chunk_text, start_char, end_char))
                    current_chunk_paras = []
                    current_tokens = 0
                
                sub_chunks = self._sentence_chunking(para)
                for sc, sc_start, sc_end in sub_chunks:
                    chunks.append((sc, sc_start, sc_end))
            elif current_tokens + para_tokens > self.chunk_size and current_chunk_paras:
                # Flush current chunk
                chunk_text = "\n\n".join(current_chunk_paras)
                idx = text.find(chunk_text, start_char)
                if idx != -1:
                    start_char = idx
                end_char = start_char + len(chunk_text)
                chunks.append((chunk_text, start_char, end_char))

                current_chunk_paras = [para]
                current_tokens = para_tokens
            else:
                current_chunk_paras.append(para)
                current_tokens += para_tokens

        if current_chunk_paras:
            chunk_text = "\n\n".join(current_chunk_paras)
            idx = text.find(chunk_text, start_char)
            if idx != -1:
                start_char = idx
            end_char = start_char + len(chunk_text)
            chunks.append((chunk_text, start_char, end_char))

        return chunks
