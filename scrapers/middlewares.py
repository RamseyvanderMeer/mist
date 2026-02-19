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
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.proxy_url = "http://proxy-server.scraperapi.com:8001"
        self.banned_domains = set()
        self.domain_failures = defaultdict(int)
        self.max_failures = 5  # Stop scraping site after 5 proxy failures

    @classmethod
    def from_crawler(cls, crawler):
        return cls(api_key=crawler.settings.get("SCRAPERAPI_KEY"))

    def _get_domain(self, url):
        return urlparse(url).netloc

    def process_request(self, request, spider):
        if not self.api_key:
            return

        domain = self._get_domain(request.url)
        if domain in self.banned_domains:
            logger.debug(f"Ignoring request to banned domain: {domain}")
            raise IgnoreRequest(f"Domain {domain} is banned due to repeated proxy failures")

        # If retry_with_proxy is set, apply proxy settings
        if request.meta.get("retry_with_proxy"):
            request.meta["proxy"] = self.proxy_url
            user_pass = f"scraperapi:{self.api_key}"
            encoded_user_pass = base64.b64encode(user_pass.encode()).decode()
            request.headers["Proxy-Authorization"] = f"Basic {encoded_user_pass}"
            request.meta["download_timeout"] = 60
            # Disable standard retries for proxy requests - we want to count them as failures immediately
            request.meta["dont_retry"] = True 

    def process_response(self, request, response, spider):
        if not self.api_key:
            return response

        # Check for blocking status codes
        if response.status in [403, 429]:
            domain = self._get_domain(request.url)
            
            # Case 1: Already using proxy and still blocked
            if request.meta.get("retry_with_proxy"):
                self.domain_failures[domain] += 1
                logger.warning(
                    f"Proxy request failed for {request.url} (Status: {response.status}). "
                    f"Failure count for {domain}: {self.domain_failures[domain]}/{self.max_failures}"
                )
                
                if self.domain_failures[domain] >= self.max_failures:
                    logger.error(f"Domain {domain} has exceeded proxy failure limit. Blocking future requests.")
                    self.banned_domains.add(domain)
                
                return response # Return response to let spider handle failure (or RetryMiddleware if configured, but we set dont_retry)
            
            # Case 2: Direct request blocked -> Retry with proxy
            else:
                if domain in self.banned_domains:
                    return response

                logger.info(f"Direct request blocked ({response.status}) for {request.url}. Retrying with proxy.")
                retry_req = request.copy()
                retry_req.meta["retry_with_proxy"] = True
                retry_req.meta["recache"] = True # Don't use cache for this retry
                retry_req.dont_filter = True # Allow duplicate request (since it's a retry)
                retry_req.priority = request.priority + 10 # Prioritize retry
                return retry_req

        # Reset failure count on success (optional, but good for transient issues)
        if response.status == 200 and request.meta.get("retry_with_proxy"):
             domain = self._get_domain(request.url)
             if self.domain_failures[domain] > 0:
                 self.domain_failures[domain] = max(0, self.domain_failures[domain] - 1)

        return response
