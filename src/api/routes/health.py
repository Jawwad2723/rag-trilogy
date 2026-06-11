from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from qdrant_client import AsyncQdrantClient
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness check."""
    return {"status": "ok"}

@router.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness check verifying connection to critical backends (like Qdrant)."""
    qdrant_url = settings.retrieval.vector_store.url or "http://localhost:6333"
    try:
        # Check connection to Qdrant
        client = AsyncQdrantClient(url=qdrant_url)
        # Simple operation to check connection
        await client.get_collections()
        await client.close()
        return {"status": "ready", "qdrant": "connected"}
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        return Response(
            content='{"status": "unready", "qdrant": "disconnected"}',
            status_code=503,
            media_type="application/json"
        ) # type: ignore

@router.get("/metrics")
def metrics() -> Response:
    """Exposes Prometheus scrape metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
