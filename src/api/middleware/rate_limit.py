import time
import asyncio
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from src.logger import get_logger

logger = get_logger(__name__)

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float) -> None:
        """
        capacity: Maximum tokens the bucket can hold.
        refill_rate: Number of tokens added to the bucket per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Refill tokens
            self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Any,
        capacity: int = 50,
        refill_rate: float = 2.0  # Refills 2 tokens per second
    ) -> None:
        super().__init__(app)
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = asyncio.Lock()

    async def _get_bucket(self, client_ip: str) -> TokenBucket:
        async with self.lock:
            if client_ip not in self.buckets:
                self.buckets[client_ip] = TokenBucket(self.capacity, self.refill_rate)
            return self.buckets[client_ip]

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Resolve client IP
        client_ip = request.client.host if request.client else "unknown-ip"
        
        # Bypass rate limit on health checks
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)

        bucket = await self._get_bucket(client_ip)
        allowed = await bucket.consume(1)

        if not allowed:
            logger.warning("rate_limit_exceeded", ip=client_ip, path=request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too Many Requests. Token bucket rate limit exceeded."}
            )

        return await call_next(request)
from typing import Any
