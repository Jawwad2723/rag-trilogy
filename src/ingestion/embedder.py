import asyncio
from abc import ABC, abstractmethod
from typing import List
import httpx
from openai import AsyncOpenAI, RateLimitError
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings and return their vector representations."""
        pass

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small", batch_size: int = 100) -> None:
        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError("OpenAI API key must be provided or set in environment.")
        self.client = AsyncOpenAI(api_key=key)
        self.model = model
        self.batch_size = batch_size

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    async def _embed_batch_with_retry(self, batch: List[str], retries: int = 5, backoff_factor: float = 0.5) -> List[List[float]]:
        for attempt in range(retries):
            try:
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                return [data.embedding for data in response.data]
            except RateLimitError as e:
                if attempt == retries - 1:
                    logger.error("openai_rate_limit_failed", error=str(e), attempt=attempt)
                    raise e
                sleep_time = backoff_factor * (2 ** attempt)
                logger.warning("openai_rate_limit_retry", sleep_time=sleep_time, attempt=attempt)
                await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error("openai_embedding_failed", error=str(e))
                raise e
        return []

class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 100) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        # Lazily load local model to avoid importing heavy weights at start
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("loading_local_sentence_transformer", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Run embedding in a separate thread so it doesn't block the event loop
        loop = asyncio.get_running_loop()
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await loop.run_in_executor(
                None, 
                lambda b: self._get_model().encode(b, convert_to_numpy=True).tolist(), 
                batch
            )
            embeddings.extend(batch_embeddings)
        return embeddings
