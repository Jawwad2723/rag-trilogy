import hashlib
import os
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
import aiofiles
from pypdf import PdfReader
from src.logger import get_logger

logger = get_logger(__name__)

class Document(BaseModel):
    text: str
    metadata: dict = Field(default_factory=dict)

def calculate_sha256(filepath: str) -> str:
    """Calculate the SHA-256 hash of a file's content."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks of 4096 bytes
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class DocumentLoader:
    def __init__(self) -> None:
        pass

    async def load_file(self, filepath: str) -> List[Document]:
        """Loads a single file (PDF, MD, HTML, TXT) and returns a list of Documents."""
        if not os.path.exists(filepath):
            logger.error("file_not_found", filepath=filepath)
            raise FileNotFoundError(f"File not found: {filepath}")

        doc_hash = calculate_sha256(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()
        base_metadata = {
            "source": os.path.abspath(filepath),
            "filename": os.path.basename(filepath),
            "ingested_at": datetime.utcnow().isoformat(),
            "doc_hash": doc_hash,
        }

        try:
            if file_ext == ".pdf":
                return self._load_pdf(filepath, base_metadata)
            elif file_ext in (".md", ".markdown"):
                return await self._load_text_async(filepath, base_metadata, "markdown")
            elif file_ext in (".html", ".htm"):
                return await self._load_text_async(filepath, base_metadata, "html")
            else:
                # Fallback to plain text loader
                return await self._load_text_async(filepath, base_metadata, "txt")
        except Exception as e:
            logger.error("file_load_failed", filepath=filepath, error=str(e))
            raise e

    def _load_pdf(self, filepath: str, base_metadata: dict) -> List[Document]:
        """Loads a PDF file page by page using pypdf."""
        documents = []
        reader = PdfReader(filepath)
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            metadata = base_metadata.copy()
            metadata["page_number"] = page_idx + 1
            metadata["total_pages"] = total_pages
            documents.append(Document(text=text, metadata=metadata))

        logger.info("pdf_loaded", filepath=filepath, pages=len(documents))
        return documents

    async def _load_text_async(self, filepath: str, base_metadata: dict, format_type: str) -> List[Document]:
        """Asynchronously loads a text-based file."""
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()

        metadata = base_metadata.copy()
        metadata["format"] = format_type
        metadata["page_number"] = 1  # Text files are treated as single page
        
        logger.info("text_loaded", filepath=filepath, format=format_type)
        return [Document(text=content, metadata=metadata)]
