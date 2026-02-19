import base64
import logging
from collections import defaultdict
from urllib.parse import urlparse
from scrapy.exceptions import IgnoreRequest

logger = logging.getLogger(__name__)

class SmartProxyMiddleware:
    """
    Middleware that routes requests through ScraperAPI only when they are blocked (403/429).
    Implements a circuit breaker to stop scraping a domain if proxy requests also fail repeatedly.
    Supports fallback keys: after 3 consecutive proxy failures with current key, switches to
    SCRAPERAPI_KEY_ONE, then SCRAPERAPI_KEY_TWO.
    """
    def __init__(self, api_keys: list[str]):
        self.api_keys = [k for k in api_keys if k]
        self.proxy_url = "http://proxy-server.scraperapi.com:8001"
        self.banned_domains = set()
        self.proxy_exhausted_domains = set()  # All keys broken; stop proxy retries, continue direct
        self.domain_failures = defaultdict(int)
        self.max_failures = 5  # Stop scraping site after 5 proxy failures (when keys still available)
        self.current_key_index = 0
        self.key_failures = 0  # Consecutive failures with current key
        self.key_rotate_threshold = 3

    @classmethod
    def from_crawler(cls, crawler):
        keys = [
            crawler.settings.get("SCRAPERAPI_KEY"),
            crawler.settings.get("SCRAPERAPI_KEY_ONE"),
            crawler.settings.get("SCRAPERAPI_KEY_TWO"),
        ]
        return cls(api_keys=keys)

    def _get_domain(self, url):
        return urlparse(url).netloc

    def _get_api_key(self):
        """Get the current API key (rotated key is used for all future requests once exhausted)."""
        idx = min(self.current_key_index, len(self.api_keys) - 1) if self.api_keys else 0
        return self.api_keys[idx] if idx < len(self.api_keys) else None

    def process_request(self, request, spider):
        if not self.api_keys:
            return

        domain = self._get_domain(request.url)
        if domain in self.banned_domains:
            logger.debug(f"Ignoring request to banned domain: {domain}")
            raise IgnoreRequest(f"Domain {domain} is banned due to repeated proxy failures")

        # If retry_with_proxy is set, apply proxy settings (always use current key index)
        if request.meta.get("retry_with_proxy"):
            api_key = self._get_api_key()
            if not api_key:
                return
            request.meta["proxy"] = self.proxy_url
            user_pass = f"scraperapi:{api_key}"
            encoded_user_pass = base64.b64encode(user_pass.encode()).decode()
            request.headers["Proxy-Authorization"] = f"Basic {encoded_user_pass}"
            request.meta["download_timeout"] = 60
            request.meta["dont_retry"] = True

    def process_response(self, request, response, spider):
        if not self.api_keys:
            return response

        # Check for blocking status codes
        if response.status in [403, 429]:
            domain = self._get_domain(request.url)

            # Case 1: Already using proxy and still blocked
            if request.meta.get("retry_with_proxy"):
                self.key_failures += 1

                # Rotate to next key after threshold; new key used for all future requests
                if (
                    self.key_failures >= self.key_rotate_threshold
                    and self.current_key_index + 1 < len(self.api_keys)
                ):
                    self.current_key_index += 1
                    self.key_failures = 0
                    logger.info(
                        f"Key exhausted. Rotating to ScraperAPI key index {self.current_key_index} "
                        f"for all future requests."
                    )
                    retry_req = request.copy()
                    retry_req.meta["retry_with_proxy"] = True
                    retry_req.meta["recache"] = True
                    retry_req.dont_filter = True
                    retry_req.priority = request.priority + 10
                    return retry_req

                # All keys exhausted: stop proxy retries, continue with direct scraping
                if self.current_key_index + 1 >= len(self.api_keys):
                    self.proxy_exhausted_domains.add(domain)
                    logger.warning(
                        f"All ScraperAPI keys exhausted for {domain}. Stopping proxy retries, "
                        f"continuing with direct requests."
                    )
                    return response

                # Under threshold: count as domain failure (may ban if keys still available)
                self.domain_failures[domain] += 1
                logger.warning(
                    f"Proxy request failed for {request.url} (Status: {response.status}). "
                    f"Failure count for {domain}: {self.domain_failures[domain]}/{self.max_failures}"
                )

                if self.domain_failures[domain] >= self.max_failures:
                    logger.error(
                        f"Domain {domain} has exceeded proxy failure limit. Blocking future requests."
                    )
                    self.banned_domains.add(domain)

                return response

            # Case 2: Direct request blocked -> Retry with proxy (unless we've given up on proxies)
            else:
                if domain in self.banned_domains:
                    return response
                if domain in self.proxy_exhausted_domains:
                    return response  # All keys broken; don't retry with proxy, continue direct

                logger.info(f"Direct request blocked ({response.status}) for {request.url}. Retrying with proxy.")
                retry_req = request.copy()
                retry_req.meta["retry_with_proxy"] = True
                retry_req.meta["recache"] = True
                retry_req.dont_filter = True
                retry_req.priority = request.priority + 10
                return retry_req

        # Reset failure counts on success
        if response.status == 200 and request.meta.get("retry_with_proxy"):
            domain = self._get_domain(request.url)
            if self.domain_failures[domain] > 0:
                self.domain_failures[domain] = max(0, self.domain_failures[domain] - 1)
            self.key_failures = 0

        return response
