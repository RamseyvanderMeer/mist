"""Authentication dependencies and middleware for FastAPI."""
import hashlib
import hmac
import logging
import os
import secrets
from typing import Optional, List
from fastapi import Request, HTTPException, Depends, status, Response
from sqlalchemy.orm import Session, joinedload
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis
import jwt
from jwt.exceptions import InvalidTokenError
import requests

from src.database.pg_connection import get_db, get_db_context
from src.models import User
from src.auth.google_oauth import (
    get_authorization_bearer,
    google_oauth_enabled,
    verify_google_id_token_string,
)

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
    """Generate rate limit key based on user or guest identity."""
    bearer = get_authorization_bearer(request)
    if bearer and google_oauth_enabled():
        claims = verify_google_id_token_string(bearer)
        if claims:
            email = (claims.get("email") or "").lower().strip()
            if email:
                return f"ratelimit:{email}"

    email = request.headers.get(IAP_EMAIL_HEADER, "")
    if email:
        if ":" in email:
            email = email.split(":")[-1]
        email = email.lower().strip()
        return f"ratelimit:{email}"

    guest_id = verify_guest_token(request.cookies.get(GUEST_COOKIE_NAME))
    if guest_id:
        return f"guest:{guest_id}"
    
    return f"ratelimit:ip:{get_remote_address(request)}"

limiter = Limiter(key_func=get_rate_limit_key)

# IAP Headers
IAP_EMAIL_HEADER = "X-Goog-Authenticated-User-Email"
IAP_SUBJECT_HEADER = "X-Goog-Authenticated-User-Id"
IAP_JWT_HEADER = "X-Goog-Iap-Jwt-Assertion"

# IAP JWT verification settings
IAP_ISSUER = "https://cloud.google.com/iap"
IAP_JWKS_URL = "https://www.gstatic.com/iap/verify/public_key-jwk"

# Development mode - skip JWT verification
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

GUEST_COOKIE_NAME = os.getenv("GUEST_COOKIE_NAME", "mist_guest")
GUEST_COOKIE_MAX_AGE = int(os.getenv("GUEST_COOKIE_MAX_AGE", str(60 * 60 * 24 * 365)))
GUEST_COOKIE_SECRET = os.getenv("GUEST_COOKIE_SECRET", "")
GUEST_REQUESTS_PER_DAY = int(os.getenv("GUEST_REQUESTS_PER_DAY", "3"))

# Cache for IAP public keys
_iap_public_keys = None

def get_iap_public_keys():
    """Fetch and cache IAP public keys."""
    global _iap_public_keys
    if _iap_public_keys is None:
        try:
            response = requests.get(IAP_JWKS_URL, timeout=10)
            response.raise_for_status()
            jwks = response.json()
            _iap_public_keys = {}
            for key in jwks.get("keys", []):
                kid = key.get("kid")
                if kid:
                    _iap_public_keys[kid] = key
        except Exception as e:
            logger.error(f"Failed to fetch IAP public keys: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service unavailable",
            )
    return _iap_public_keys


def verify_iap_jwt(request: Request) -> dict:
    """
    Verify the IAP JWT token.
    Returns the decoded payload if valid.
    Raises HTTPException if invalid.
    """
    # Debug: log all headers
    logger.debug(f"All headers: {dict(request.headers)}")
    
    jwt_token = request.headers.get(IAP_JWT_HEADER)
    
    if not jwt_token:
        # Debug: check if header exists with different case
        for key in request.headers.keys():
            if "jwt" in key.lower():
                logger.warning(f"Found JWT-related header: {key} = {request.headers.get(key)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing IAP JWT token. Access denied. Looking for header: {IAP_JWT_HEADER}",
        )
    
    try:
        # Get the key ID from the token header
        unverified_header = jwt.get_unverified_header(jwt_token)
        kid = unverified_header.get("kid")
        
        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid JWT token: missing key ID",
            )
        
        # Get the public key
        public_keys = get_iap_public_keys()
        if kid not in public_keys:
            # Refresh keys if not found
            global _iap_public_keys
            _iap_public_keys = None
            public_keys = get_iap_public_keys()
            if kid not in public_keys:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid JWT token: unknown key ID",
                )
        
        public_key = public_keys[kid]
        
        # Verify the token
        payload = jwt.decode(
            jwt_token,
            key=public_key,
            algorithms=["ES256"],
            issuer=IAP_ISSUER,
            audience="/projects/33176763769/global/backendServices/33176763769",  # Update this
        )
        
        return payload
        
    except InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT token: {str(e)}",
        )
    except Exception as e:
        logger.error(f"JWT verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication verification failed",
        )


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


def _guest_secret_bytes() -> bytes:
    secret = GUEST_COOKIE_SECRET.strip()
    if not secret:
        logger.warning("GUEST_COOKIE_SECRET not set; using volatile process secret for guest cookies")
        secret = f"volatile-{os.getpid()}"
    return secret.encode("utf-8")


def sign_guest_token(guest_id: str) -> str:
    sig = hmac.new(_guest_secret_bytes(), guest_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{guest_id}.{sig}"


def verify_guest_token(token: str | None) -> Optional[str]:
    if not token or "." not in token:
        return None
    guest_id, sig = token.rsplit(".", 1)
    if not guest_id or not sig:
        return None
    expected = hmac.new(_guest_secret_bytes(), guest_id.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    return guest_id


def issue_guest_cookie(response: Response, guest_id: Optional[str] = None) -> str:
    guest_id = guest_id or secrets.token_urlsafe(24)
    token = sign_guest_token(guest_id)
    response.set_cookie(
        key=GUEST_COOKIE_NAME,
        value=token,
        max_age=GUEST_COOKIE_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
    )
    return guest_id


def get_or_create_guest_id(request: Request, response: Optional[Response] = None) -> str:
    token = request.cookies.get(GUEST_COOKIE_NAME)
    guest_id = verify_guest_token(token)
    if guest_id:
        return guest_id
    if response is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Guest session missing. Start a guest session first.",
        )
    return issue_guest_cookie(response)


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
    """
    Get current user from IAP JWT token (required).
    Verifies the JWT signature before trusting headers.
    In DEV_MODE, skips JWT verification and uses headers directly.
    """
    # Development mode: skip JWT verification
    if DEV_MODE:
        logger.warning("DEV_MODE enabled: Skipping JWT verification")
        email = get_iap_email(request)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="DEV_MODE: X-Goog-Authenticated-User-Email header required",
            )
    else:
        # Production: IAP JWT (header) and/or Google OAuth ID token (Authorization: Bearer)
        jwt_header = request.headers.get(IAP_JWT_HEADER)
        bearer = get_authorization_bearer(request)
        google_claims = None
        if bearer and google_oauth_enabled():
            google_claims = verify_google_id_token_string(bearer)

        email = None
        if jwt_header:
            try:
                payload = verify_iap_jwt(request)
            except HTTPException:
                raise
            email = (payload.get("email") or "").lower()
            if not email:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid JWT: missing email claim",
                )
            header_email = get_iap_email(request)
            if header_email and header_email != email:
                logger.warning("IAP JWT email mismatch: JWT=%s, Header=%s", email, header_email)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Email mismatch between JWT and headers",
                )
        elif google_claims:
            email = (google_claims.get("email") or "").lower()
            if not email:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Google token missing email claim",
                )
            header_email = get_iap_email(request)
            if header_email and header_email != email:
                logger.warning("Google token email mismatch: token=%s, Header=%s", email, header_email)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Email mismatch between Google token and headers",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Authentication required: send X-Goog-Iap-Jwt-Assertion (IAP) or "
                    "Authorization: Bearer <Google ID token> when GOOGLE_OAUTH_CLIENT_IDS is set."
                ),
            )
    
    user = db.query(User).filter(User.email == email.lower()).first()
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
    
    # Explicitly blocked users cannot use the API
    if user.tier and user.tier.requests_per_minute == 0 and user.tier.requests_per_hour == 0 and user.tier.requests_per_day == 0:
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
        return "1/day"  # Any authenticated user gets a minimal default allowance
    
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


def get_guest_rate_limit() -> str:
    return f"{GUEST_REQUESTS_PER_DAY}/day"


def tier_limit_for_ratelimit_key(key: str) -> str:
    """
    Dynamic slowapi limit: must accept a parameter named ``key`` (see slowapi LimitGroup).

    ``key`` is the value from ``get_rate_limit_key(request)`` (email-, guest-, or IP-based).
    """
    guest_prefix = "guest:"
    if key.startswith(guest_prefix):
        return get_guest_rate_limit()

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


class GuestIdentity:
    def __init__(self, guest_id: str):
        self.id = guest_id
        self.email = f"guest:{guest_id}"
        self.display_name = "Guest"
        self.status = "guest"
        self.roles = []
        self.tier = None


async def get_current_actor(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    bearer = get_authorization_bearer(request)
    has_google = bool(bearer and google_oauth_enabled())
    has_iap = bool(request.headers.get(IAP_JWT_HEADER))
    has_dev_email = bool(DEV_MODE and get_iap_email(request))
    if has_google or has_iap or has_dev_email:
        return await get_current_user(request, db)
    guest_id = get_or_create_guest_id(request, response)
    return GuestIdentity(guest_id)


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
