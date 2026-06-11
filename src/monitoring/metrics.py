from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

# We define standard buckets in seconds:
# 50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s, 10s
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

# Latency histograms
rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent performing retrieval (vector + BM25 + fusion)",
    buckets=LATENCY_BUCKETS
)

rag_reranking_latency_seconds = Histogram(
    "rag_reranking_latency_seconds",
    "Time spent reranking hybrid retrieval results",
    buckets=LATENCY_BUCKETS
)

rag_generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Time spent generating response via LLM",
    buckets=LATENCY_BUCKETS
)

rag_total_latency_seconds = Histogram(
    "rag_total_latency_seconds",
    "Total RAG end-to-end request latency",
    buckets=LATENCY_BUCKETS
)

# Cost and token counters
rag_cost_usd_total = Counter(
    "rag_cost_usd_total",
    "Total cost of RAG requests in USD",
    labelnames=["model"]
)

rag_tokens_total = Counter(
    "rag_tokens_total",
    "Total tokens consumed by RAG generation",
    labelnames=["model", "type"]  # type: "prompt" or "completion"
)

# Quality counters
rag_citation_violations_total = Counter(
    "rag_citation_violations_total",
    "Total number of citation validation failures"
)

rag_no_answer_total = Counter(
    "rag_no_answer_total",
    "Total responses where LLM returned a refusal (e.g. 'I don't have enough info')"
)

# Throughput and system state
rag_requests_total = Counter(
    "rag_requests_total",
    "Total requests received by RAG API",
    labelnames=["model", "status"]
)

rag_requests_in_flight = Gauge(
    "rag_requests_in_flight",
    "Number of in-flight RAG API requests currently processing"
)
