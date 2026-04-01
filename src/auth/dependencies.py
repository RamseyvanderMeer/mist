"""Authentication dependencies and middleware for FastAPI."""
import logging
import os
from typing import Optional, List
from fastapi import Request, HTTPException, Depends, status
from sqlalchemy.orm import Session, joinedload
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis

from src.database.pg_connection import get_db, get_db_context
from src.models import User

logger = logging.getLogger(__name__)

# Redis connection for rate limiting
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Initialize Redis client
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
except Exception as e:
    print(f"Warning: Redis connection failed: {e}")
    redis_client = None

# Initialize rate limiter with custom key function
def get_rate_limit_key(request: Request) -> str:
    """Generate rate limit key based on user identity."""
    # Try to get user email from IAP header
    email = request.headers.get("X-Goog-Authenticated-User-Email", "")
    if email:
        # Remove 'accounts.google.com:' prefix if present
        if ":" in email:
            email = email.split(":")[-1]
        email = email.lower().strip()
        return f"ratelimit:{email}"
    
    # Fallback to IP address
    return f"ratelimit:ip:{get_remote_address(request)}"

limiter = Limiter(key_func=get_rate_limit_key)

# IAP Headers
IAP_EMAIL_HEADER = "X-Goog-Authenticated-User-Email"
IAP_SUBJECT_HEADER = "X-Goog-Authenticated-User-Id"


def get_iap_email(request: Request) -> Optional[str]:
    """Extract and clean email from IAP headers."""
    email = request.headers.get(IAP_EMAIL_HEADER)
    if not email:
        return None
    
    # Remove 'accounts.google.com:' or 'google.com:' prefix
    if ":" in email:
        email = email.split(":")[-1]
    
    return email.lower().strip()


def get_iap_subject(request: Request) -> Optional[str]:
    """Extract subject ID from IAP headers."""
    subject = request.headers.get(IAP_SUBJECT_HEADER)
    if not subject:
        return None
    
    # Remove 'accounts.google.com:' prefix if present
    if ":" in subject:
        subject = subject.split(":")[-1]
    
    return subject


async def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user from IAP headers if available (optional)."""
    email = get_iap_email(request)
    if not email:
        return None
    
    user = db.query(User).filter(User.email == email).first()
    return user


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
) -> User:
    """Get current user from IAP headers (required)."""
    email = get_iap_email(request)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in via Google.",
        )
    
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not registered. Please complete registration.",
        )
    
    if not user.is_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is suspended. Contact support.",
        )
    
    # Check tier - blocked users (0 requests) cannot use API
    if user.tier and user.tier.requests_per_minute == 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account tier does not have API access. Please upgrade your plan.",
        )
    
    # Update last login
    from datetime import datetime
    user.last_login_at = datetime.utcnow()
    db.commit()
    
    return user


def require_roles(required_roles: List[str]):
    """Dependency factory to require specific roles."""
    async def role_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if not current_user.has_any_role(required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires one of the following roles: {', '.join(required_roles)}",
            )
        return current_user
    return role_checker


require_admin = require_roles(["admin"])
require_user = require_roles(["user", "admin"])


def get_tier_rate_limit(user: User) -> str:
    """Get rate limit string for user's tier."""
    if not user.tier:
        return "0/minute"  # Blocked by default
    
    rpm = user.tier.requests_per_minute
    rph = user.tier.requests_per_hour
    rpd = user.tier.requests_per_day
    
    limits = []
    if rpm > 0:
        limits.append(f"{rpm}/minute")
    if rph > 0:
        limits.append(f"{rph}/hour")
    if rpd > 0:
        limits.append(f"{rpd}/day")
    
    if not limits:
        return "0/minute"  # Blocked
    
    return ",".join(limits)


def tier_limit_for_ratelimit_key(key: str) -> str:
    """
    Dynamic slowapi limit: must accept a parameter named ``key`` (see slowapi LimitGroup).

    ``key`` is the value from ``get_rate_limit_key(request)`` (email- or IP-based).
    """
    ip_prefix = "ratelimit:ip:"
    if key.startswith(ip_prefix):
        return os.getenv("RATE_LIMIT_IP_FALLBACK", "1000/minute")

    email_prefix = "ratelimit:"
    if not key.startswith(email_prefix):
        return "0/minute"

    email = key[len(email_prefix) :].strip().lower()
    if not email:
        return "0/minute"

    try:
        with get_db_context() as db:
            user = (
                db.query(User)
                .options(joinedload(User.tier))
                .filter(User.email == email)
                .first()
            )
    except Exception:
        logger.exception("tier_limit_for_ratelimit_key: DB error for key=%r", key)
        return "0/minute"

    if not user:
        return "0/minute"

    return get_tier_rate_limit(user)


class RateLimitMiddleware:
    """Custom rate limiting middleware with tier support."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Rate limiting handled by SlowAPI decorator on endpoints
        await self.app(scope, receive, send)
