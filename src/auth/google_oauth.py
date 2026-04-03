"""Verify Google OAuth ID tokens (Sign in with Google) for mobile / SPA clients."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import Request

logger = logging.getLogger(__name__)

_google_request = None


def _google_transport_request():
    global _google_request
    if _google_request is None:
        from google.auth.transport import requests as google_auth_requests

        _google_request = google_auth_requests.Request()
    return _google_request


def google_oauth_client_ids() -> list[str]:
    raw = os.getenv("GOOGLE_OAUTH_CLIENT_IDS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def google_oauth_enabled() -> bool:
    return bool(google_oauth_client_ids())


def get_authorization_bearer(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization") or ""
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        return token or None
    return None


def verify_google_id_token_string(token: str) -> Optional[dict[str, Any]]:
    """
    Validate a Google OAuth ID token (iss accounts.google.com).
    Tries each configured OAuth client ID as audience until verification succeeds.
    """
    allowed = google_oauth_client_ids()
    if not allowed or not token:
        return None
    try:
        from google.oauth2 import id_token as google_id_token
    except ImportError:
        logger.error("google-oauth: google-auth not installed; pip install google-auth")
        return None

    req = _google_transport_request()
    for aud in allowed:
        try:
            return google_id_token.verify_oauth2_token(token, req, aud)
        except Exception as e:
            logger.debug("Google ID token verify failed for aud=%s: %s", aud, e)
            continue
    return None


def identity_from_google_bearer(request: Request) -> Optional[dict[str, Any]]:
    """Return verified Google token claims if Authorization Bearer is a valid Google ID token."""
    bearer = get_authorization_bearer(request)
    if not bearer:
        return None
    return verify_google_id_token_string(bearer)
