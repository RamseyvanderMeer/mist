# Scraper Discovery: New Sources

Analysis of potential new data sources for MIST.

## 1. Bimmerpost Family
**Sites**:
- `https://f30.bimmerpost.com/forums/`
- `https://g60.bimmerpost.com/forums/`
- `https://www.2addicts.com/forums/`

**Platform**: vBulletin 3.8

**Status**: **Mixed / Fragile**
- Search requires POST to `search.php?do=process`.
- Simple scripts often timeout or get blocked.
- **Recommendation**: Do not use search-based scraping for Bimmerpost. Instead, crawl specific forum sections (e.g., `/forums/forumdisplay.php?f=421` for F30 Maintenance) and filter threads by title/content.

## 2. Bimmer Owners Club
**Site**: `https://www.bimmerownersclub.com/forums/`

**Platform**: Invision Community (IP.Board)

**Status**: **Validated**
- Search URL: `https://www.bimmerownersclub.com/search/?q={query}&type=forums_topic`
- Pagination: Standard Invision pagination.
- **Recommendation**: Add to scraper using `FORUM_CONFIGS`.

## 3. Reddit (r/BmwTech)
**Site**: `https://www.reddit.com/r/BmwTech/`

**Status**: **Validated**
- API URL: `https://www.reddit.com/r/BmwTech/search.json?q={query}&restrict_sr=1`
- **Recommendation**: Add a dedicated `RedditSpider` or extend `ForumSpider` to handle JSON responses for Reddit.

## 4. OEM DTC
**Site**: `https://bmw.oemdtc.com/`

**Status**: **Blocked (403)**
- Returns 403 Forbidden to scripts.
- **Recommendation**: Skip for now.

## 5. Other Sites (Skipped)
- **Bimmerforums**: Cloudflare blocked.
- **BMWFaultCodes**: CAPTCHA.
- **NHTSA**: Use API if needed.
- **Facebook**: Requires auth.
- **BMW MOA**: Likely vBulletin, but lower priority given Bimmerpost coverage.

## Implementation Plan
1. Add **Bimmer Owners Club** to `FORUM_CONFIGS` in `scrapers/utils/forum_config.py`.
2. Create a Reddit-specific handling in `ForumSpider` or `RedditSpider`.
