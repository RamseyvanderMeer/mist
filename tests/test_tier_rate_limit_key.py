"""Unit tests for slowapi tier-based dynamic rate limit provider."""
from unittest.mock import MagicMock, patch

import pytest

from src.auth.dependencies import get_tier_rate_limit, tier_limit_for_ratelimit_key


def _user_with_tier(rpm=10, rph=100, rpd=500):
    tier = MagicMock()
    tier.requests_per_minute = rpm
    tier.requests_per_hour = rph
    tier.requests_per_day = rpd
    user = MagicMock()
    user.tier = tier
    return user


class TestGetTierRateLimit:
    def test_composite_limits(self):
        u = _user_with_tier(10, 100, 500)
        assert get_tier_rate_limit(u) == "10/minute,100/hour,500/day"

    def test_no_tier_blocked(self):
        u = MagicMock()
        u.tier = None
        assert get_tier_rate_limit(u) == "0/minute"

    def test_zero_tier_blocked(self):
        u = _user_with_tier(0, 0, 0)
        assert get_tier_rate_limit(u) == "0/minute"


class TestTierLimitForRatelimitKey:
    @patch.dict("os.environ", {"RATE_LIMIT_IP_FALLBACK": "50/minute"}, clear=False)
    def test_ip_key_uses_env_fallback(self):
        assert tier_limit_for_ratelimit_key("ratelimit:ip:127.0.0.1") == "50/minute"

    @patch("src.auth.dependencies.get_db_context")
    def test_email_key_user_found(self, mock_ctx):
        mock_db = MagicMock()
        user = _user_with_tier(5, 0, 0)
        mock_db.query.return_value.options.return_value.filter.return_value.first.return_value = (
            user
        )
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_db
        mock_cm.__exit__.return_value = None
        mock_ctx.return_value = mock_cm

        assert tier_limit_for_ratelimit_key("ratelimit:alice@example.com") == "5/minute"
        mock_db.query.assert_called_once()

    @patch("src.auth.dependencies.get_db_context")
    def test_email_key_user_not_found(self, mock_ctx):
        mock_db = MagicMock()
        mock_db.query.return_value.options.return_value.filter.return_value.first.return_value = None
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_db
        mock_cm.__exit__.return_value = None
        mock_ctx.return_value = mock_cm

        assert tier_limit_for_ratelimit_key("ratelimit:nobody@example.com") == "0/minute"

    @patch("src.auth.dependencies.get_db_context")
    def test_db_error_returns_blocked(self, mock_ctx):
        mock_cm = MagicMock()
        mock_cm.__enter__.side_effect = RuntimeError("db down")
        mock_ctx.return_value = mock_cm

        assert tier_limit_for_ratelimit_key("ratelimit:any@example.com") == "0/minute"

    def test_malformed_key_blocked(self):
        assert tier_limit_for_ratelimit_key("not-a-key") == "0/minute"


@pytest.mark.parametrize(
    "header_email,expected_prefix",
    [
        ("accounts.google.com:User@Example.COM", "ratelimit:user@example.com"),
    ],
)
def test_get_rate_limit_key_normalizes_email(header_email, expected_prefix):
    from unittest.mock import MagicMock

    from src.auth.dependencies import get_rate_limit_key

    req = MagicMock()
    req.headers = {"X-Goog-Authenticated-User-Email": header_email}
    assert get_rate_limit_key(req) == expected_prefix
