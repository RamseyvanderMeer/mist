"""
Rate limiting middleware for MIST API.
Prevents abuse and controls costs.
"""
from fastapi import Request, HTTPException
from fastapi.middleware import Middleware
import time
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter.
    For production, use Redis or Cloud Memorystore.
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed for client.
        Returns (allowed, headers).
        """
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                ts for ts in self.requests[client_id] 
                if ts > window_start
            ]
        else:
            self.requests[client_id] = []
        
        # Check rate limit
        current_count = len(self.requests[client_id])
        
        if current_count >= self.requests_per_minute:
            # Rate limit exceeded
            retry_after = int(60 - (now - self.requests[client_id][0]))
            return False, {
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(self.requests[client_id][0] + 60)),
                "Retry-After": str(max(1, retry_after))
            }
        
        # Allow request
        self.requests[client_id].append(now)
        remaining = self.requests_per_minute - current_count - 1
        
        return True, {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(now + 60))
        }


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=60,  # 60 requests per minute per client
    burst_size=10            # Allow burst of 10
)


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware for rate limiting.
    """
    # Get client identifier (IP or API key)
    client_id = request.headers.get("X-API-Key") or request.client.host
    
    # Check rate limit
    allowed, headers = rate_limiter.is_allowed(client_id)
    
    if not allowed:
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers=headers
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers to response
    for header, value in headers.items():
        response.headers[header] = value
    
    return response


class APIKeyValidator:
    """
    API key validation middleware.
    """
    
    def __init__(self, valid_keys: list = None):
        self.valid_keys = set(valid_keys or [])
        self.require_key = bool(valid_keys)
    
    async def __call__(self, request: Request, call_next):
        """Validate API key if required."""
        if not self.require_key:
            return await call_next(request)
        
        # Skip validation for health endpoint
        if request.url.path == "/health":
            return await call_next(request)
        
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            logger.warning(f"Missing API key from {request.client.host}")
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Include X-API-Key header."
            )
        
        if api_key not in self.valid_keys:
            logger.warning(f"Invalid API key from {request.client.host}")
            raise HTTPException(
                status_code=403,
                detail="Invalid API key."
            )
        
        return await call_next(request)


def setup_security(app, api_keys: list = None):
    """
    Setup security middleware for FastAPI app.
    
    Args:
        app: FastAPI application
        api_keys: List of valid API keys (optional)
    """
    # Add rate limiting
    app.middleware("http")(rate_limit_middleware)
    
    # Add API key validation if keys provided
    if api_keys:
        validator = APIKeyValidator(api_keys)
        app.middleware("http")(validator)
        logger.info(f"API key validation enabled with {len(api_keys)} keys")
    
    logger.info("Security middleware configured")
