from fastapi import FastAPI
from src.config import settings
from src.logger import configure_logger, get_logger
from src.monitoring.tracer import configure_tracer
from src.api.middleware.tracing import TracingMiddleware
from src.api.middleware.rate_limit import RateLimiterMiddleware
from src.api.routes import query, ingest, health
from src.api.deps import get_vector_store, get_bm25_store

# Initialize logging before anything else
configure_logger(settings.app.log_level)
logger = get_logger(__name__)

# Initialize tracing exporter
configure_tracer(service_name=settings.app.name)

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="Production-Ready Monitored Hybrid RAG API"
)

# Attach Middlewares
app.add_middleware(TracingMiddleware)
app.add_middleware(RateLimiterMiddleware, capacity=50, refill_rate=2.0)

# Include Routers
app.include_router(query.router, tags=["Query"])
app.include_router(ingest.router, tags=["Ingest"])
app.include_router(health.router, tags=["System Health"])

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("api_server_startup", app_name=settings.app.name, version=settings.app.version)
    # Proactively warm up stores
    try:
        await get_vector_store()._get_client()
        logger.info("vector_store_warmed_up")
    except Exception as e:
        logger.warning("vector_store_warmup_failed_non_blocking", error=str(e))

@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("api_server_shutdown_initiated")
    # Clean up Qdrant client connection
    try:
        await get_vector_store().close()
    except Exception as e:
        logger.error("error_closing_vector_store", error=str(e))

    # Persist BM25 index to disk
    try:
        await get_bm25_store().persist()
    except Exception as e:
        logger.error("error_persisting_bm25_store", error=str(e))

    logger.info("api_server_shutdown_completed")
