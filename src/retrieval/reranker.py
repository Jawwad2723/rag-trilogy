import asyncio
import time
from typing import List
from sentence_transformers import CrossEncoder
from opentelemetry import trace
from src.config import settings
from src.retrieval.vector_store import SearchResult
from src.logger import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("loading_reranker_model", model=self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(self, query: str, results: List[SearchResult], top_n: int = 5) -> List[SearchResult]:
        """Reranks the search results using CrossEncoder. Returns top_n results."""
        if not self.enabled or not results:
            return results[:top_n]

        start_time = time.time()
        
        # Start Otel span for reranking
        with tracer.start_as_current_span("rag.rerank") as span:
            span.set_attribute("rag.rerank.model", self.model_name)
            span.set_attribute("rag.rerank.input_count", len(results))

            try:
                loop = asyncio.get_running_loop()
                
                # Prepare inputs for the cross-encoder: List of [query, text]
                pairs = [[query, res.text] for res in results]

                # Run blocking inference in a thread pool
                def predict() -> List[float]:
                    model = self._get_model()
                    return model.predict(pairs).tolist() # type: ignore

                scores = await loop.run_in_executor(None, predict)

                # Pair results with scores and sort
                scored_results = []
                for res, score in zip(results, scores):
                    # Create copy of result with updated score
                    scored_results.append(
                        SearchResult(
                            chunk_id=res.chunk_id,
                            text=res.text,
                            metadata=res.metadata,
                            score=float(score)
                        )
                    )

                # Sort by score descending
                scored_results.sort(key=lambda x: x.score, reverse=True)
                final_results = scored_results[:top_n]

                latency_ms = (time.time() - start_time) * 1000
                span.set_attribute("rag.rerank.output_count", len(final_results))
                span.set_attribute("rag.rerank.latency_ms", latency_ms)
                
                logger.info("reranking_success", latency_ms=latency_ms, inputs=len(results), outputs=len(final_results))
                return final_results

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.warning("reranking_failed_fallback", error=str(e), latency_ms=latency_ms)
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                # Fallback to returning original top_n results without reranking
                return results[:top_n]
