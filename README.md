# rag-trilogy 🔍

> **Production-grade, fully observable RAG system** — hybrid retrieval, local SLMs, end-to-end monitoring.

[![CI Pipeline](https://github.com/your-org/rag-trilogy/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rag-trilogy/actions/workflows/ci.yml)
[![Quality Gate](https://github.com/your-org/rag-trilogy/actions/workflows/eval_gate.yml/badge.svg)](https://github.com/your-org/rag-trilogy/actions/workflows/eval_gate.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

---

## Overview

`rag-trilogy` is a **monorepo** that implements three tightly integrated layers of a production RAG system:

| Layer | Description |
|-------|-------------|
| **Project 1 — Ask My Docs** | Core retrieval + generation pipeline with citation enforcement |
| **Project 2 — Local SLM (Ollama)** | Drop-in local model backend; zero external API calls |
| **Project 3 — Monitoring & Observability** | Prometheus metrics, Grafana dashboards, OTEL traces, RAGAS CI gate |

---

## Architecture

```
                          ┌────────────────────────────────────────────┐
                          │               FastAPI Server                │
                          │  ┌─────────┐  ┌──────────┐  ┌──────────┐  │
  HTTP Request ──────────►│  │ /query  │  │ /ingest  │  │ /metrics │  │
                          │  └────┬────┘  └────┬─────┘  └──────────┘  │
                          │       │             │                        │
                          │  ┌────▼────────────▼──────────────────┐    │
                          │  │          RAGChain                   │    │
                          │  │  ┌─────────────┐  ┌─────────────┐  │    │
                          │  │  │   Retrieve  │  │  Generate   │  │    │
                          │  │  └──────┬──────┘  └──────┬──────┘  │    │
                          │  └─────────┼─────────────────┼─────────┘   │
                          └────────────┼─────────────────┼─────────────┘
                                       │                 │
              ┌────────────────────────▼──┐    ┌─────────▼─────────────┐
              │       HybridRetriever     │    │      LLM Client        │
              │                           │    │                        │
              │  ┌──────────┬──────────┐  │    │ ┌──────┬────┬───────┐ │
              │  │  Qdrant  │  BM25   │  │    │ │OpenAI│Ant │Ollama │ │
              │  │ (Vector) │(Keyword)│  │    │ └──────┴────┴───────┘ │
              │  └────┬─────┴────┬────┘  │    └────────────────────────┘
              │       └────┬─────┘       │
              │      ┌─────▼─────┐       │
              │      │   RRF     │       │
              │      │  Fusion   │       │
              │      └─────┬─────┘       │
              │      ┌─────▼─────┐       │
              │      │CrossEncoder│       │
              │      │ Reranker  │       │
              │      └───────────┘       │
              └───────────────────────────┘
```

### Key Design Decisions

- **No LangChain/LlamaIndex** — Built on raw primitives for full production transparency and debuggability
- **Parallel retrieval** — BM25 and vector search are equal primary signals, not fallbacks
- **Citation enforcement** — Every factual claim must have a `[chunk_id]` citation; violations are tracked and alerted
- **Swap-in local SLMs** — Replace any OpenAI call with `model: "ollama/llama3.2"` at query time
- **Observable by default** — Every request generates OTEL spans, Prometheus counters, and structured JSON logs

---

## Monorepo Structure

```
rag-trilogy/
├── src/
│   ├── config.py                   # Pydantic settings + YAML loader
│   ├── logger.py                   # Structured JSON logging (structlog)
│   ├── ingestion/
│   │   ├── loader.py               # PDF/MD/HTML/TXT document loader (SHA-256 dedup)
│   │   ├── chunker.py              # Fixed / sentence / semantic chunking
│   │   ├── embedder.py             # OpenAI & sentence-transformers embedders
│   │   └── pipeline.py             # Ingestion orchestrator
│   ├── retrieval/
│   │   ├── vector_store.py         # Async Qdrant client (HNSW, on-disk payload)
│   │   ├── bm25_store.py           # Persistent BM25 index (pickle)
│   │   ├── hybrid.py               # RRF fusion retriever
│   │   └── reranker.py             # CrossEncoder reranker (async thread)
│   ├── generation/
│   │   ├── llm_client.py           # OpenAI / Anthropic / Ollama clients (streaming)
│   │   ├── prompt_templates.py     # Citation-enforcing system prompt
│   │   └── chain.py                # Full RAG chain with metrics + OTEL
│   ├── monitoring/
│   │   ├── tracer.py               # OTEL trace provider (OTLP → Jaeger)
│   │   ├── metrics.py              # Prometheus counters / histograms / gauges
│   │   ├── cost_tracker.py         # Per-model token cost calculator
│   │   └── quality_metrics.py      # RAGAS evaluation interface
│   ├── evaluation/
│   │   ├── dataset.py              # Golden dataset I/O
│   │   ├── ragas_eval.py           # Batch RAGAS evaluator
│   │   ├── regression_gate.py      # Quality threshold CI gate
│   │   └── benchmark.py            # Multi-model latency + quality benchmarker
│   └── api/
│       ├── main.py                 # FastAPI app with lifespan events
│       ├── deps.py                 # Dependency injection (singletons)
│       ├── routes/
│       │   ├── query.py            # POST /query (sync + SSE streaming)
│       │   ├── ingest.py           # POST /ingest (file upload)
│       │   └── health.py           # GET /health, /ready, /metrics
│       └── middleware/
│           ├── tracing.py          # OTEL request span middleware
│           └── rate_limit.py       # Token bucket rate limiter
├── monitoring/
│   ├── prometheus.yml              # Prometheus scrape config
│   ├── alerts/
│   │   └── quality_regression.yml  # Alert: citation violation > 5%
│   └── grafana/
│       ├── datasources.yml
│       ├── provisioning_dashboards.yml
│       └── dashboards/
│           ├── rag_overview.json   # Latency, throughput, in-flight
│           ├── cost_tracker.json   # Cost per model, token splits
│           └── quality_drift.json  # Citation violations, refusal rate
├── tests/
│   ├── unit/                       # Pure unit tests (no I/O)
│   ├── integration/                # FastAPI TestClient + mocked backends
│   └── eval/
│       └── golden_dataset.json     # 50 curated Q&A pairs (factual/reasoning/adversarial)
├── scripts/
│   ├── ingest_docs.py              # CLI: bulk ingest a directory
│   ├── run_eval.py                 # CLI: RAGAS evaluation suite
│   └── benchmark_models.py        # CLI: compare Ollama models
├── .github/workflows/
│   ├── ci.yml                      # Lint + type-check + test on every PR
│   └── eval_gate.yml               # Weekly RAGAS quality regression gate
├── docker-compose.yml              # Full stack (app + Qdrant + Prometheus + Grafana + Jaeger)
├── docker-compose.local.yml        # Local Ollama variant (no cloud API calls)
├── Dockerfile
├── config.yaml                     # Main system configuration
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) [Ollama](https://ollama.ai) for local models

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env and set your API keys
```

### 3. Run Full Stack

```bash
# Cloud LLM mode (OpenAI / Anthropic)
docker compose up -d

# Local SLM mode (Ollama, no external API)
docker compose -f docker-compose.local.yml up -d
```

Services available at:
| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Grafana | http://localhost:3000 (admin/admin) |
| Jaeger | http://localhost:16686 |
| Prometheus | http://localhost:9091 |
| Qdrant | http://localhost:6333/dashboard |

### 4. Install Local Dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Usage

### Ingest Documents

```bash
# Via CLI (ingest all docs in a directory)
python scripts/ingest_docs.py --dir ./data/docs/

# Via API
curl -X POST http://localhost:8000/ingest \
  -F "file=@./data/docs/my_document.pdf"

# Via Makefile
make ingest DIR=./data/docs/
```

### Query Documents

```bash
# Sync query (OpenAI GPT-4o-mini)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the retrieval strategy?", "model": "default"}'

# Local model (Ollama)
curl -X POST http://localhost:8000/query \
  -d '{"question": "Summarize the ingestion pipeline.", "model": "ollama/llama3.2"}'

# Streaming response (SSE)
curl -N -X POST http://localhost:8000/query \
  -d '{"question": "Explain hybrid search.", "stream": true}'

# Keyword-only retrieval (alpha=0.0)
curl -X POST http://localhost:8000/query \
  -d '{"question": "BM25 algorithm", "alpha": 0.0}'
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | str | required | The question to ask |
| `top_k` | int | 5 | Number of context chunks to retrieve |
| `alpha` | float | 0.5 | Hybrid weight: 0=BM25 only, 1=vector only |
| `model` | str | `"default"` | `gpt-4o-mini`, `claude-3-haiku-20240307`, `ollama/llama3.2` |
| `stream` | bool | false | Enable SSE streaming response |
| `filters` | dict | null | Metadata filters e.g. `{"filename": "guide.md"}` |

---

## Local SLM Layer (Ollama)

The system supports a **full zero-egress mode** using Ollama:

```bash
# Pull a model locally
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull phi3:mini

# Run with local stack
docker compose -f docker-compose.local.yml up -d

# Query using local model
curl -X POST http://localhost:8000/query \
  -d '{"question": "What is RAG?", "model": "ollama/mistral:7b"}'
```

### Benchmarking Local Models

```bash
# Compare llama3.2:3b, mistral:7b, phi3:mini on 20 questions
make bench

# Or directly
python scripts/benchmark_models.py --questions 20
```

Sample benchmark output:

| Model | Avg t/s | p50 lat | p95 lat | Faithfulness | Context Recall | Verdict |
|-------|---------|---------|---------|--------------|----------------|---------|
| llama3.2:3b | 58.2 | 1.1s | 2.4s | 0.79 | 0.72 | Fast/OK |
| mistral:7b | 24.1 | 2.8s | 5.2s | 0.87 | 0.81 | Balanced |
| phi3:mini | 71.4 | 0.9s | 1.8s | 0.74 | 0.68 | Good trade |

---

## Monitoring & Observability

### Metrics (Prometheus → Grafana)

| Metric | Type | Description |
|--------|------|-------------|
| `rag_total_latency_seconds` | Histogram | End-to-end request latency |
| `rag_retrieval_latency_seconds` | Histogram | Time spent in hybrid retrieval |
| `rag_generation_latency_seconds` | Histogram | Time spent in LLM generation |
| `rag_cost_usd_total` | Counter | Cumulative cost by model |
| `rag_tokens_total` | Counter | Prompt + completion token counts |
| `rag_citation_violations_total` | Counter | Citation enforcement failures |
| `rag_requests_total` | Counter | Requests by model and status |
| `rag_requests_in_flight` | Gauge | Concurrent active requests |

Three pre-built Grafana dashboards are provisioned automatically:
- **RAG Overview** — Latency, throughput, in-flight requests
- **Cost Tracker** — Spending by model, token splits, cost rate
- **Quality Drift** — Citation violation rate, refusal rate over time

### Traces (OTEL → Jaeger)

Every request creates a root span `rag.query` with child spans:
- `rag.retrieve.hybrid` — RRF fusion timing
- `rag.rerank` — CrossEncoder timing
- `rag.generate` — LLM generation timing

Each span carries attributes: model, cost, tokens, citation violations.

---

## Evaluation & CI Quality Gate

### Run RAGAS Evaluation

```bash
make eval
# Or:
python scripts/run_eval.py --limit 20 --output eval_results.json
```

### Quality Thresholds

| Metric | Threshold |
|--------|-----------|
| Faithfulness | ≥ 0.80 |
| Answer Relevancy | ≥ 0.75 |
| Context Precision | ≥ 0.70 |
| Citation Violation Rate | ≤ 5% |

### CI Pipeline

The `eval_gate.yml` GitHub Actions workflow:
1. Runs every Monday at 6am UTC and on changes to `src/retrieval/` or `src/generation/`
2. Executes the full RAGAS evaluation suite against the golden dataset
3. Calls `regression_gate.py` which exits with code 1 on any threshold breach
4. Blocks the PR/merge if quality regresses

---

## Development

### Common Commands

```bash
# Run tests with coverage
make test

# Type checking
make typecheck

# Security audit
make audit

# Ingest documents
make ingest DIR=./data/

# Run evaluation
make eval

# Benchmark Ollama models
make bench

# Full stack (cloud)
make up

# Local Ollama stack
make up-local

# Stop services
make down
```

### Running Tests Locally

```bash
source .venv/bin/activate
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Type Checking

```bash
mypy --strict src/ --ignore-missing-imports
```

### Security Audit

```bash
bandit -r src/ -ll
```

---

## Configuration Reference

All configuration is in `config.yaml` with environment variable overrides.

```yaml
retrieval:
  embedding:
    provider: openai          # openai | local
    model: text-embedding-3-small
    dimensions: 1536
  hybrid:
    alpha: 0.5               # 0=BM25, 1=vector, 0.5=balanced
    rrf_k: 60                # RRF constant (higher = flatter ranking)
  reranker:
    enabled: true
    model: cross-encoder/ms-marco-MiniLM-L-6-v2

generation:
  default_provider: openai   # openai | anthropic | ollama
  default_model: gpt-4o-mini
  max_tokens: 1024
  temperature: 0.1
  ollama:
    base_url: ${OLLAMA_BASE_URL}
    default_model: llama3.2:3b
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Cloud mode | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic mode | Anthropic API key |
| `QDRANT_URL` | No | Qdrant URL (default: `http://localhost:6333`) |
| `OLLAMA_BASE_URL` | Local mode | Ollama server URL (default: `http://localhost:11434`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | OTEL collector (default: `http://localhost:4317`) |

---

## Citation Enforcement

The system enforces strict citation rules on every generated answer:

```
CITATION RULES:
1. Every factual statement MUST be followed by [chunk_id] citation
2. If answer cannot be found in context: respond ONLY with the refusal message
3. NEVER use prior knowledge — only what is in the provided document context
```

Violations are:
- Logged as structured warnings
- Counted in `rag_citation_violations_total` Prometheus counter
- Gated in CI (>5% violation rate fails the pipeline)
- Alertable via Prometheus `HighCitationViolations` alert rule

---

## Performance Targets

| Metric | Target |
|--------|--------|
| p50 E2E latency | < 1.5s |
| p95 E2E latency | < 3.0s |
| Retrieval latency | < 100ms |
| Reranking latency | < 250ms |
| Throughput | > 50 RPS |

---

## License

MIT — See [LICENSE](LICENSE) for details.
