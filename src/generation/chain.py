import hashlib
import re
import time
from typing import AsyncGenerator, Dict, List, Any, Tuple, Optional
from opentelemetry import trace
from src.config import settings
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import SearchResult
from src.generation.llm_client import BaseLLMClient, OpenAIClient, AnthropicClient, OllamaClient, LLMResponse
from src.generation.prompt_templates import CITATION_SYSTEM_PROMPT
from src.monitoring.cost_tracker import CostTracker
from src.monitoring.metrics import (
    rag_retrieval_latency_seconds,
    rag_reranking_latency_seconds,
    rag_generation_latency_seconds,
    rag_total_latency_seconds,
    rag_cost_usd_total,
    rag_tokens_total,
    rag_citation_violations_total,
    rag_no_answer_total,
    rag_requests_total,
    rag_requests_in_flight
)
from src.logger import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

class RAGChain:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        cost_tracker: CostTracker,
        clients: Dict[str, BaseLLMClient] | None = None
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.cost_tracker = cost_tracker
        
        # Initialize LLM Clients
        self.clients = clients or {
            "openai": OpenAIClient(default_model=settings.generation.default_model),
            "anthropic": AnthropicClient(default_model="claude-3-haiku-20240307"),
            "ollama": OllamaClient(default_model=settings.generation.ollama.default_model)
        }

    def _get_client_for_model(self, model: str) -> Tuple[BaseLLMClient, str]:
        """Maps a model name to the appropriate LLM client and updates the model string if necessary."""
        if model == "default":
            provider = settings.generation.default_provider
            model_name = settings.generation.default_model
        elif model.startswith("ollama/"):
            provider = "ollama"
            model_name = model.replace("ollama/", "")
        elif model.startswith("claude-") or "haiku" in model:
            provider = "anthropic"
            model_name = model
        else:
            # Default fallback to openai
            provider = "openai"
            model_name = model

        client = self.clients.get(provider)
        if not client:
            raise ValueError(f"LLM Client provider '{provider}' is not configured.")
        return client, model_name

    def _check_citation_violation(self, text: str) -> bool:
        """
        Regex-based validation for citation rules:
        If the response contains >2 sentences without any citation, flag it.
        Refusals ("I don't have enough information...") are excluded.
        """
        refusal_msg = "I don't have enough information in the provided documents to answer this question."
        if refusal_msg.lower() in text.lower() or "enough information" in text.lower():
            return False

        # Split sentences via punctuation
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
        
        # Citation markers: [chunk_id], [SOURCE: file, page N], [chunk_xxx]
        # Match SHA-256 hashes inside brackets, e.g. [58c9fa...] or [SOURCE: ...] or [chunk_...]
        citation_pattern = re.compile(r"(\[[a-f0-9]{32,64}\]|\[SOURCE:[^\]]+\]|\[chunk_[a-f0-9]+\])")
        citations = citation_pattern.findall(text)

        if len(sentences) > 2 and len(citations) == 0:
            return True
        return False

    async def query(
        self,
        question: str,
        top_k: int = 5,
        alpha: float = 0.5,
        model: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Executes the complete RAG flow: retrieve -> rerank -> prompt -> LLM completion."""
        rag_requests_in_flight.inc()
        start_total = time.time()
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()

        # Start OTEL Root Span
        with tracer.start_as_current_span("rag.query") as root_span:
            root_span.set_attribute("rag.question_hash", question_hash)
            root_span.set_attribute("rag.alpha", alpha)

            # 1. Retrieval
            start_retrieval = time.time()
            with tracer.start_as_current_span("rag.retrieve.hybrid") as retrieve_span:
                retrieve_span.set_attribute("rag.retrieve.top_k", top_k)
                retrieved_chunks = await self.retriever.retrieve(
                    query=question,
                    top_k=top_k * 4,  # Retrieve more for reranking candidate pool
                    alpha=alpha,
                    filters=filters
                )
            retrieval_latency = (time.time() - start_retrieval) * 1000
            rag_retrieval_latency_seconds.observe(retrieval_latency / 1000.0)

            # 2. Rerank
            start_reranking = time.time()
            reranked_chunks = await self.reranker.rerank(
                query=question,
                results=retrieved_chunks,
                top_n=top_k
            )
            reranking_latency = (time.time() - start_reranking) * 1000
            rag_reranking_latency_seconds.observe(reranking_latency / 1000.0)

            # Map to context text block and format citations references
            context_blocks = []
            citations = []
            for rank, chunk in enumerate(reranked_chunks):
                source = chunk.metadata.get("filename", "unknown")
                page = chunk.metadata.get("page_number", 1)
                context_blocks.append(
                    f"Chunk ID: {chunk.chunk_id}\n"
                    f"Source: {source}\n"
                    f"Page: {page}\n"
                    f"Content: {chunk.text}\n"
                    "---"
                )
                citations.append({
                    "chunk_id": chunk.chunk_id,
                    "source": source,
                    "page": page,
                    "text_preview": chunk.text[:150]
                })

            context_str = "\n".join(context_blocks)

            # 3. LLM Generation
            client, resolved_model = self._get_client_for_model(model)
            full_model_name = f"ollama/{resolved_model}" if isinstance(client, OllamaClient) else resolved_model
            
            root_span.set_attribute("rag.model", full_model_name)
            root_span.set_attribute("rag.chunks_retrieved", len(reranked_chunks))

            messages = [
                {
                    "role": "system",
                    "content": CITATION_SYSTEM_PROMPT.format(context=context_str, question=question)
                },
                {
                    "role": "user",
                    "content": question
                }
            ]

            start_gen = time.time()
            with tracer.start_as_current_span("rag.generate") as gen_span:
                gen_span.set_attribute("rag.generate.model", full_model_name)
                
                llm_response = await client.complete(
                    messages=messages,
                    temperature=settings.generation.temperature,
                    max_tokens=settings.generation.max_tokens,
                    model=resolved_model
                )
                
                gen_span.set_attribute("rag.generate.prompt_tokens", llm_response.prompt_tokens)
                gen_span.set_attribute("rag.generate.completion_tokens", llm_response.completion_tokens)
                gen_span.set_attribute("rag.generate.latency_ms", llm_response.latency_ms)

            gen_latency = (time.time() - start_gen) * 1000
            rag_generation_latency_seconds.observe(gen_latency / 1000.0)

            # 4. Post-generation quality & citation validation
            citation_violation = self._check_citation_violation(llm_response.text)
            root_span.set_attribute("rag.citation_violations", citation_violation)

            # Cost Tracking
            cost = self.cost_tracker.calculate_cost(
                model=full_model_name,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens
            )
            root_span.set_attribute("rag.estimated_cost_usd", cost)

            # Prometheus metrics updates
            rag_cost_usd_total.labels(model=full_model_name).inc(cost)
            rag_tokens_total.labels(model=full_model_name, type="prompt").inc(llm_response.prompt_tokens)
            rag_tokens_total.labels(model=full_model_name, type="completion").inc(llm_response.completion_tokens)
            
            if citation_violation:
                rag_citation_violations_total.inc()
                logger.warning("citation_enforcement_violation", question_hash=question_hash, text=llm_response.text)

            is_refusal = "I don't have enough information" in llm_response.text
            if is_refusal:
                rag_no_answer_total.inc()

            total_latency = (time.time() - start_total) * 1000
            rag_total_latency_seconds.observe(total_latency / 1000.0)
            
            status_code = "success"
            rag_requests_total.labels(model=full_model_name, status=status_code).inc()
            rag_requests_in_flight.dec()

            response_data = {
                "answer": llm_response.text,
                "citations": citations if not is_refusal else [],
                "metadata": {
                    "retrieval_latency_ms": round(retrieval_latency, 2),
                    "reranking_latency_ms": round(reranking_latency, 2),
                    "generation_latency_ms": round(gen_latency, 2),
                    "total_latency_ms": round(total_latency, 2),
                    "model_used": full_model_name,
                    "prompt_tokens": llm_response.prompt_tokens,
                    "completion_tokens": llm_response.completion_tokens,
                    "estimated_cost_usd": round(cost, 6),
                    "chunks_retrieved": len(reranked_chunks),
                    "citation_violations": citation_violation
                }
            }

            logger.info("rag_query_completed", **response_data["metadata"])
            return response_data

    async def stream_query(
        self,
        question: str,
        top_k: int = 5,
        alpha: float = 0.5,
        model: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streams the RAG response tokens, finishing with a metadata block."""
        rag_requests_in_flight.inc()
        start_total = time.time()
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()

        # 1. Retrieval
        start_retrieval = time.time()
        retrieved_chunks = await self.retriever.retrieve(
            query=question,
            top_k=top_k * 4,
            alpha=alpha,
            filters=filters
        )
        retrieval_latency = (time.time() - start_retrieval) * 1000
        rag_retrieval_latency_seconds.observe(retrieval_latency / 1000.0)

        # 2. Rerank
        start_reranking = time.time()
        reranked_chunks = await self.reranker.rerank(
            query=question,
            results=retrieved_chunks,
            top_n=top_k
        )
        reranking_latency = (time.time() - start_reranking) * 1000
        rag_reranking_latency_seconds.observe(reranking_latency / 1000.0)

        context_blocks = []
        citations = []
        for rank, chunk in enumerate(reranked_chunks):
            source = chunk.metadata.get("filename", "unknown")
            page = chunk.metadata.get("page_number", 1)
            context_blocks.append(
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {chunk.text}\n"
                "---"
            )
            citations.append({
                "chunk_id": chunk.chunk_id,
                "source": source,
                "page": page,
                "text_preview": chunk.text[:150]
            })

        context_str = "\n".join(context_blocks)

        client, resolved_model = self._get_client_for_model(model)
        full_model_name = f"ollama/{resolved_model}" if isinstance(client, OllamaClient) else resolved_model

        messages = [
            {
                "role": "system",
                "content": CITATION_SYSTEM_PROMPT.format(context=context_str, question=question)
            },
            {
                "role": "user",
                "content": question
            }
        ]

        # 3. LLM Stream
        full_text = []
        prompt_tokens = 0
        completion_tokens = 0
        gen_latency = 0.0

        try:
            async for chunk in client.stream(
                messages=messages,
                temperature=settings.generation.temperature,
                max_tokens=settings.generation.max_tokens,
                model=resolved_model
            ):
                token = chunk.get("token", "")
                if token:
                    full_text.append(token)
                    yield {"token": token}
                
                # Check for metadata yielded at the very end
                if "metadata" in chunk:
                    meta = chunk["metadata"]
                    prompt_tokens = meta.get("prompt_tokens", 0)
                    completion_tokens = meta.get("completion_tokens", 0)
                    gen_latency = meta.get("latency_ms", 0.0)

            full_response_text = "".join(full_text)
            rag_generation_latency_seconds.observe(gen_latency / 1000.0)

            # 4. Citations & Metrics
            citation_violation = self._check_citation_violation(full_response_text)
            cost = self.cost_tracker.calculate_cost(
                model=full_model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )

            # Prometheus updates
            rag_cost_usd_total.labels(model=full_model_name).inc(cost)
            rag_tokens_total.labels(model=full_model_name, type="prompt").inc(prompt_tokens)
            rag_tokens_total.labels(model=full_model_name, type="completion").inc(completion_tokens)
            if citation_violation:
                rag_citation_violations_total.inc()
            
            is_refusal = "I don't have enough information" in full_response_text
            if is_refusal:
                rag_no_answer_total.inc()

            total_latency = (time.time() - start_total) * 1000
            rag_total_latency_seconds.observe(total_latency / 1000.0)
            rag_requests_total.labels(model=full_model_name, status="success").inc()

            yield {
                "metadata": {
                    "citations": citations if not is_refusal else [],
                    "retrieval_latency_ms": round(retrieval_latency, 2),
                    "reranking_latency_ms": round(reranking_latency, 2),
                    "generation_latency_ms": round(gen_latency, 2),
                    "total_latency_ms": round(total_latency, 2),
                    "model_used": full_model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "estimated_cost_usd": round(cost, 6),
                    "chunks_retrieved": len(reranked_chunks),
                    "citation_violations": citation_violation
                }
            }

        except Exception as e:
            rag_requests_total.labels(model=full_model_name, status="error").inc()
            logger.error("rag_stream_query_failed", error=str(e))
            raise e
        finally:
            rag_requests_in_flight.dec()
