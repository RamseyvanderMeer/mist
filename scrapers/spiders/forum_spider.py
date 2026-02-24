"""
Forum spider for automotive diagnostic data.

Scrapes BimmerFest (XenForo), E90Post (vBulletin), and similar forums for fault codes,
OBD data, vehicle context, and repair summaries.

Supports: search-based discovery, fault-code search per forum, pagination, skip-already-parsed.
Tracks which (forum, code) search combinations have been completed for resumable --search-codes runs.
"""
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import scrapy

from scrapers.spiders.base import MistBaseSpider, merge_text
from scrapers.utils.forum_config import FAULT_CODES_TO_SEARCH, FORUM_CONFIGS

logger = logging.getLogger(__name__)

SEARCH_PROGRESS_FILENAME = "search_progress.jsonl"

# Fault code pattern for title filtering
FAULT_CODE_PATTERN = re.compile(r"\b(P[0-9]{4}|[0-9A-Z]{4,5})\b", re.IGNORECASE)
# Keywords suggesting fault/diagnostic content
TITLE_KEYWORDS = {
    "check engine", "cel", "fault code", "obd", "code", "diagnostic",
    "misfire", "error", "dtc", "trouble code", "engine light", "abs",
    "transmission", "limp mode", "stalling", "rough idle",
    # Expanded keywords
    "symptom", "problem", "issue", "fail", "failing", "broken", "bad",
    "wont start", "no start", "crank", "cranking",
    "noise", "rattle", "whine", "clunk", "leak", "smoke", "overheat",
    "vibration", "shake", "shaking", "hesitate", "hesitation",
    "power loss", "loss of power", "sluggish", "surge", "surging",
    "rough run", "running rough", "jerk", "jerking",
    "warning", "message", "malfunction",
}
# Keywords suggesting a confirmed fix (prioritize these - higher likelihood of extractable data)
TITLE_FIX_KEYWORDS = {
    "fixed", "solved", "replaced", "installed", "repaired", "cleaned",
    "rebuilt", "fixed it", "that fixed", "fixed the", "problem solved",
    "issue resolved", "fixed my", "replacement", "fix for",
    # Expanded keywords
    "solution", "success", "update", "finally", "sorted", "resolved",
    "changed", "swapped", "new part",
    # Gratitude (often indicates a solution was provided)
    "thanks", "thank you", "appreciated", "cheers", "kudos",
    # Technical fix/procedure keywords
    "diy", "guide", "tutorial", "how to", "how-to", "retrofit", "coding",
    "programmed", "flashed", "registered", "calibrated", "adapted",
    "reset", "procedure", "instructions",
}


def _extract_search_forum_code(url: str) -> tuple[str, str] | None:
    """Extract (forum_name, fault_code) from a search URL, or None if not a search URL."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        forum = None
        if "bimmerfest.com" in netloc:
            forum = "bimmerfest"
        elif "bimmerownersclub.com" in netloc:
            forum = "bimmerownersclub"
        elif "oemdtc.com" in netloc:
            forum = "oemdtc"
        if not forum:
            return None
        qs = parse_qs(parsed.query)
        # BimmerFest: keywords=P0300
        if "keywords" in qs and qs["keywords"]:
            return (forum, qs["keywords"][0].strip().upper())
        # BimmerOwnersClub: q=P0300
        if "q" in qs and qs["q"]:
            return (forum, qs["q"][0].strip().upper())
        # OEMDTC: s=P0300
        if "s" in qs and qs["s"]:
            return (forum, qs["s"][0].strip().upper())
    except Exception:
        pass
    return None


def _load_searched_codes(output_dir: Path) -> set[tuple[str, str]]:
    """Load (forum, code) pairs that have already been searched."""
    seen = set()
    path = Path(output_dir) / SEARCH_PROGRESS_FILENAME
    if not path.exists():
        return seen
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    forum = rec.get("forum")
                    code = rec.get("code")
                    if forum and code:
                        seen.add((forum, code.upper()))
                except json.JSONDecodeError:
                    continue
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Could not load search progress: %s", e)
    return seen


def _save_searched_code(output_dir: Path, forum: str, code: str) -> None:
    """Append a (forum, code) pair to search progress file."""
    path = Path(output_dir) / SEARCH_PROGRESS_FILENAME
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"forum": forum, "code": code.upper()}) + "\n")
    except OSError as e:
        logger.warning("Could not save search progress: %s", e)


def _build_start_urls(
    use_search: bool,
    use_targeted: bool,
    use_search_codes: bool,
    searched_codes: set[tuple[str, str]] | None = None,
) -> list[str]:
    """Build start URLs from forum configs (interleaved for round-robin)."""
    from itertools import zip_longest

    searched_codes = searched_codes or set()

    if use_search_codes:
        lists_of_urls = []
        for name, cfg in FORUM_CONFIGS.items():
            if cfg.get("supports_search") and cfg.get("search_url"):
                forum_urls = [
                    cfg["search_url"].format(code=code)
                    for code in FAULT_CODES_TO_SEARCH
                    if (name, code.upper()) not in searched_codes
                ]
                lists_of_urls.append(forum_urls)
        
        # Interleave URLs: [ForumA-1, ForumB-1, ForumA-2, ForumB-2, ...]
        if lists_of_urls:
            return [u for group in zip_longest(*lists_of_urls) for u in group if u]
            
    # Fallback to forum listing URLs (interleaved)
    lists_of_urls = []
    for name, cfg in FORUM_CONFIGS.items():
        lists_of_urls.append(cfg["forum_urls"])
        
    if use_search or use_targeted:
        return [u for group in zip_longest(*lists_of_urls) for u in group if u]

    # Default: E90Post + BimmerFest main
    return [
        "https://www.e90post.com/forums/forumdisplay.php?f=2",
        "https://www.bimmerfest.com/forums/engine-drivetrain.28/",
    ]


def _normalize_url_for_dedup(url: str) -> str:
    """Normalize URL for dedup (strip fragments, session params, extract thread id)."""
    if not url:
        return ""
    url = url.split("#")[0].rstrip("/").strip()
    if "showthread" in url or "viewtopic" in url:
        m = re.search(r"[?&]t=(\d+)", url, re.IGNORECASE)
        if m:
            return f"thread_t_{m.group(1)}"
    # XenForo: /threads/title.12345/ or /threads/title.12345/post-67890
    if "threads/" in url:
        m = re.search(r"threads/[^/]+\.(\d+)(?:/|$)", url)
        if m:
            return f"thread_t_{m.group(1)}"
    return url


def _load_seen_urls(output_dir: Path) -> set[str]:
    """Load already-parsed source URLs from Postgres when DATABASE_URL is set."""
    seen = set()
    output_dir = Path(output_dir)

    def add_url(u: str) -> None:
        if u:
            seen.add(_normalize_url_for_dedup(u))

    db_url = os.environ.get("DATABASE_URL")
    if db_url and db_url.startswith("postgresql"):
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT source_url FROM scraped_records"))
                for row in result:
                    add_url(row[0])
        except Exception as e:
            logger.debug("Could not load seen URLs from Postgres: %s", e)

    return seen


def _title_suggests_fault_content(title: str) -> bool:
    """
    True if thread title suggests fault codes, diagnostics, or a confirmed fix.
    Tightened to favor threads likely to have extractable repair data.
    """
    if not title or len(title) < 5:
        return False
    t = title.lower().strip()
    # Skip pagination links (1, 2, 3, Last, Next, etc.)
    if t.isdigit() or t in ("last", "next", "prev", "previous", "page"):
        return False
    # Strong signal: fault code in title
    if FAULT_CODE_PATTERN.search(t):
        return True
    # Strong signal: fix-related keywords (suggests resolution)
    if any(kw in t for kw in TITLE_FIX_KEYWORDS):
        return True
    # Weaker signal: diagnostic/symptom keywords (may be unanswered)
    return any(kw in t for kw in TITLE_KEYWORDS)


def _is_pagination_or_nav_link(text: str) -> bool:
    """True if link text looks like pagination/nav, not a thread title."""
    if not text:
        return True
    if len(text) > 80:
        return False  # Long text is likely a real title
    t = text.strip().lower()
    if t.isdigit():
        return True
    if t in ("last", "next", "prev", "previous", "page", "»", "«", "..."):
        return True
    if t.startswith("page ") or t.endswith(" page"):
        return True
    return False


class ForumSpider(MistBaseSpider):
    """
    Spider for automotive forums with search-based discovery and skip-already-parsed.
    """

    name = "forum"
    allowed_domains = [
        "bimmerforums.com",
        "bimmerfest.com",
        "e90post.com",
        "bimmerpost.com",
        "reddit.com",
        "old.reddit.com",
        "bimmerownersclub.com",
        "oemdtc.com",
    ]
    start_urls = []  # Set in __init__ via _build_start_urls

    def __init__(
        self,
        start_url=None,
        use_search=False,
        use_targeted=False,
        use_search_codes=False,
        re_scrape=False,
        output_dir=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._seen_urls: set[str] = set()
        self._output_dir = Path(output_dir) if output_dir else None

        if output_dir and not re_scrape:
            self._seen_urls = _load_seen_urls(Path(output_dir))
            logger.info("Loaded %d already-parsed URLs to skip", len(self._seen_urls))
        elif re_scrape:
            logger.info("Re-scrape enabled: ignoring previously seen URLs")

        if start_url:
            self.start_urls = [start_url]
        else:
            searched_codes = _load_searched_codes(self._output_dir or Path(".")) if use_search_codes else set()
            if searched_codes:
                logger.info("Skipping %d already-searched (forum, code) pairs", len(searched_codes))
            self.start_urls = _build_start_urls(use_search, use_targeted, use_search_codes, searched_codes)
            if use_search_codes:
                logger.info("Using fault-code search (%d URLs)", len(self.start_urls))
            elif use_search or use_targeted:
                logger.info("Using targeted subforums (%d URLs)", len(self.start_urls))

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        output_dir = Path(crawler.settings.get("MIST_RAW_DATA_DIR", "data/training/raw_data"))
        if spider._output_dir is None:
            spider._output_dir = output_dir
        spider._seen_urls = _load_seen_urls(output_dir)
        if spider._seen_urls:
            logger.info("Loaded %d already-parsed URLs to skip", len(spider._seen_urls))
        return spider

    def start_requests(self):
        for url in self.start_urls:
            # Direct thread URL
            if "showthread" in url or "viewtopic" in url or ("threads/" in url and "/post-" in url):
                if self._normalize_url(url) in self._seen_urls:
                    continue
                yield scrapy.Request(url, callback=self.parse_thread, dont_filter=True)
            # Search results page (BimmerFest /search/search?keywords=...)
            elif "/search" in url and "keywords=" in url:
                yield scrapy.Request(url, callback=self.parse_xenforo_search, dont_filter=True)
            # Forum listing
            else:
                yield scrapy.Request(url, callback=self._route_parse, dont_filter=True)

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for dedup (delegates to module-level)."""
        return _normalize_url_for_dedup(url)

    def _route_parse(self, response):
        """Route forum listing to XenForo or vBulletin parser."""
        url = response.url
        if "bimmerfest.com" in url:
            return self.parse_xenforo_forum(response)
        if "bimmerownersclub.com" in url:
            return self.parse_invision_forum(response)
        if "oemdtc.com" in url:
            return self.parse_wordpress_blog(response)
        return self.parse(response)

    def parse_wordpress_blog(self, response):
        """Parse OEMDTC (WordPress) blog listing or search results."""
        # Record search progress when this is a search results page
        if self._output_dir:
            extracted = _extract_search_forum_code(response.url)
            if extracted:
                forum, code = extracted
                _save_searched_code(self._output_dir, forum, code)
        # Find article links
        for href in response.css("h2.entry-title a, .post-title a"):
            url = response.urljoin(href.attrib.get("href", ""))
            title = " ".join(href.css("::text").getall() or []).strip()
            if not url:
                continue
            norm = self._normalize_url(url)
            if norm in self._seen_urls:
                continue
            # OEMDTC is dedicated to codes, so we accept almost all links, but check title for relevance
            if _title_suggests_fault_content(title) or "dtc" in url or "code" in url:
                self._seen_urls.add(norm)
                yield response.follow(url, self.parse_thread, dont_filter=True)
                
        # Pagination
        next_sel = response.css(".nav-links a.next::attr(href), .pagination a.next::attr(href)").get()
        if next_sel:
            yield response.follow(next_sel, self.parse_wordpress_blog, dont_filter=True)

    def parse_invision_forum(self, response):
        """Parse Invision Community forum listing or search results."""
        # Record search progress when this is a search results page
        if self._output_dir:
            extracted = _extract_search_forum_code(response.url)
            if extracted:
                forum, code = extracted
                _save_searched_code(self._output_dir, forum, code)
        # Thread links: a[href*='/topic/']
        for href in response.css("a[href*='/topic/']"):
            url = response.urljoin(href.attrib.get("href", ""))
            # Strip query params/fragments
            url = url.split("?")[0].split("#")[0]
            title = " ".join(href.css("::text").getall() or []).strip()
            if not url or "/topic/" not in url:
                continue
            norm = self._normalize_url(url)
            if norm in self._seen_urls:
                continue
            if _title_suggests_fault_content(title):
                self._seen_urls.add(norm)
                yield response.follow(url, self.parse_thread, dont_filter=True)
        
        # Pagination: a[rel='next']
        next_sel = response.css("a[rel='next']::attr(href)").get()
        if next_sel:
            yield response.follow(next_sel, self.parse_invision_forum, dont_filter=True)

    def parse_xenforo_search(self, response):
        """Parse BimmerFest search results - extract thread links."""
        # Record search progress (BimmerFest search URLs always have keywords=)
        if self._output_dir:
            extracted = _extract_search_forum_code(response.url)
            if extracted:
                forum, code = extracted
                _save_searched_code(self._output_dir, forum, code)
        # Links: a[href*='threads/'] - exclude /post- fragment for base thread URL
        for href in response.css("a[href*='threads/']"):
            url = response.urljoin(href.attrib.get("href", ""))
            # Strip /post-12345 to get canonical thread URL
            if "/post-" in url:
                url = re.sub(r"/post-\d+.*$", "/", url).rstrip("/") + "/"
            title = " ".join(href.css("::text").getall() or []).strip()
            if not url or "threads/" not in url:
                continue
            norm = self._normalize_url(url)
            if norm in self._seen_urls:
                continue
            if _title_suggests_fault_content(title) or "threads/" in url:
                self._seen_urls.add(norm)
                yield response.follow(url, self.parse_thread, dont_filter=True)
        # Pagination: next page of search results
        next_sel = response.css("a.pageNav-jump--next::attr(href), .block-outer a[rel='next']::attr(href)").get()
        if next_sel:
            yield response.follow(next_sel, self.parse_xenforo_search, dont_filter=True)

    def parse_xenforo_forum(self, response):
        """Parse XenForo forum listing (BimmerFest) - threads + pagination."""
        for href in response.css("a[href*='threads/']"):
            url = response.urljoin(href.attrib.get("href", ""))
            if "/post-" in url:
                url = re.sub(r"/post-\d+.*$", "/", url).rstrip("/") + "/"
            title = " ".join(href.css("::text").getall() or []).strip()
            if not url or "threads/" not in url:
                continue
            if _is_pagination_or_nav_link(title):
                continue
            norm = self._normalize_url(url)
            if norm in self._seen_urls:
                continue
            if _title_suggests_fault_content(title):
                self._seen_urls.add(norm)
                yield response.follow(url, self.parse_thread, dont_filter=True)
        # XenForo pagination: /forum.28/page-2
        next_sel = response.css("a.pageNav-jump--next::attr(href), a[rel='next']::attr(href)").get()
        if next_sel:
            yield response.follow(next_sel, self.parse_xenforo_forum, dont_filter=True)

    def parse_search_results(self, response):
        """Parse search result page - extract thread links."""
        # vBulletin search results
        for href in response.css("a[href*='showthread'], a[href*='viewtopic'], a[href*='threads/']"):
            url = href.attrib.get("href")
            title = (href.css("::text").getall() or [""])[0] if href else ""
            if url and _title_suggests_fault_content(title):
                norm = self._normalize_url(response.urljoin(url))
                if norm not in self._seen_urls:
                    yield response.follow(url, self.parse_thread)
                else:
                    logger.debug("Skip seen: %s", norm[:60])

        # Reddit search
        for href in response.css("a[data-click-id='body'], a[href*='/comments/']"):
            url = href.attrib.get("href")
            title = (href.css("h3::text, .title::text").get() or "").strip()
            if url and "comments" in url and _title_suggests_fault_content(title):
                norm = self._normalize_url(response.urljoin(url))
                if norm not in self._seen_urls:
                    self._seen_urls.add(norm)
                    yield response.follow(url, self.parse_thread)
                else:
                    logger.debug("Skip seen: %s", norm[:60])

        # Pagination
        next_sel = response.css("a.next::attr(href), .pagination a[rel='next']::attr(href), a[rel='next']::attr(href)")
        if next_sel:
            yield response.follow(next_sel[0], self.parse_search_results, dont_filter=True)

    def parse(self, response):
        """Parse forum listing - follow thread links with title filtering."""
        for href in response.css(
            "a[href*='showthread'], a[href*='viewtopic'], a[href*='threads/']"
        ):
            url = href.attrib.get("href")
            title = " ".join(href.css("::text").getall() or []).strip()
            if not url:
                continue
            if _is_pagination_or_nav_link(title):
                continue
            norm = self._normalize_url(response.urljoin(url))
            if norm in self._seen_urls:
                logger.debug("Skip seen: %s", norm[:60])
                continue
            if _title_suggests_fault_content(title):
                self._seen_urls.add(norm)  # Mark queued to avoid duplicate requests
                yield response.follow(url, self.parse_thread)
            else:
                logger.debug("Skip (no fault keywords): %s", title[:50] if title else url[:50])

        # vBulletin pagination: a.next, or numbered page links
        next_page = response.css("a.next::attr(href), .pagination a[rel='next']::attr(href)").get()
        if not next_page:
            # vBulletin often uses rel=next on page number links
            next_page = response.xpath("//a[contains(@rel,'next')]/@href").get()
        if not next_page:
            # Fallback: find "Next" or "»" link
            for a in response.css("a[href*='forumdisplay'], a[href*='page=']"):
                text = " ".join(a.css("::text").getall() or []).strip().lower()
                if text in ("next", "»", "›"):
                    next_page = a.attrib.get("href")
                    break
        if next_page:
            yield response.follow(next_page, self.parse, dont_filter=True)

    def parse_thread(self, response):
        """Parse thread page - extract content for LLM analysis."""
        # Note: We don't check seen_urls here - we only queue URLs not in seen_urls,
        # and we add to seen_urls when we queue, so we never process duplicates.
        # vBulletin: .block-body, .postbody; XenForo: .message-body, .bbWrapper; Invision: .ipsType_normal
        content_blocks = response.css(
            "main, article, [role='main'], .block-body, .postbody, .message-body, .bbWrapper, .ipsType_normal, .cPost_contentWrap"
        )
        if not content_blocks:
            content_blocks = response.css("body")

        texts = []
        for block in content_blocks:
            parts = block.css("::text").getall()
            if parts:
                texts.append(" ".join(t.strip() for t in parts if t.strip()))

        raw_text = merge_text(*texts)
        if not raw_text or len(raw_text) < 100:
            logger.debug("Thread too short or empty: %s", response.url[:80])
            return

        item = self.build_item(
            raw_text=raw_text,
            source_url=response.url,
            source_type="forum",
        )
        item["raw_text"] = raw_text
        yield item
