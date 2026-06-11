import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
import structlog
from opentelemetry import trace
from opentelemetry.trace import SpanKind

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate or capture request_id
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Set structlog contextvars for the current request context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Skip OTEL root span tracking for health/prometheus metrics endpoints to prevent noise
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)

        # Start OTEL request span
        span_name = f"HTTP {request.method} {request.url.path}"
        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.SERVER,
            attributes={
                "http.method": request.method,
                "http.target": request.url.path,
                "request_id": request_id,
                "http.client_ip": request.client.host if request.client else "unknown"
            }
        ) as span:
            response = await call_next(request)
            
            # Record response metadata in tracing
            span.set_attribute("http.status_code", response.status_code)
            return response
