# Proxy API Setup for BMWFault Scraping

## Problem
- Tor scraping: Too slow (9+ min per request due to rate limits)
- Direct scraping: IP blocks and rate limits
- Need: Fast, reliable IP rotation

## Solution: Proxy API Services

### Option 1: DataImpulse (Recommended)
**Website:** https://dataimpulse.com

**Pricing:**
- Residential proxies: $1 per GB
- Rotating IPs: Automatic
- No monthly commitment

**Setup:**
```bash
# Sign up and get proxy URL
# Format: http://user:pass@proxy.dataimpulse.com:823
# Add to .env:
DATAIMPULSE_PROXY=http://user:pass@proxy.dataimpulse.com:823
```

**Pros:**
- Cheapest option ($1/GB)
- Automatic IP rotation
- Residential IPs (less likely to be blocked)
- Pay-as-you-go

**Estimated usage:** ~500MB for 1,924 P-codes = ~$0.50

---

### Option 2: ScraperAPI
**Website:** https://www.scraperapi.com

**Pricing:**
- Free tier: 5,000 API calls/month
- Paid: $49/month for 100,000 calls

**Setup:**
```bash
# Sign up and get API key
# Add to .env:
SCRAPERAPI_KEY=your_api_key_here
```

**Pros:**
- Automatic IP rotation
- Handles CAPTCHAs
- Fast (no delays needed)
- Good success rate

---

### Option 2: ScrapingBee
**Website:** https://www.scrapingbee.com

**Pricing:**
- Free tier: 1,000 API calls
- Paid: $49/month for 100,000 calls

**Setup:**
```bash
# Add to .env:
SCRAPINGBEE_KEY=your_api_key_here
```

**Pros:**
- JavaScript rendering support
- Good documentation
- Reliable

---

### Option 3: BrightData (Formerly Luminati)
**Website:** https://brightdata.com

**Pricing:**
- Pay-per-GB: ~$15/GB
- Or monthly plans

**Setup:**
```bash
# Add to .env:
BRIGHTDATA_PROXY=http://user:pass@proxy.brightdata.com:22225
```

**Pros:**
- Largest proxy network
- Very reliable
- Residential IPs

---

## Usage

### With DataImpulse (Recommended):
```bash
# Set environment variable
export DATAIMPULSE_PROXY=http://user:pass@proxy.dataimpulse.com:823

# Run scraper
cd /mnt/external_ssd/mist
python3 scripts/fetch_bmwfault_proxyapi.py
```

### With ScraperAPI:
```bash
# Set environment variable
export SCRAPERAPI_KEY=your_key_here

# Run scraper
cd /mnt/external_ssd/mist
python3 scripts/fetch_bmwfault_proxyapi.py
```

## Estimated Cost

For 1,924 P-codes:
- **DataImpulse: ~$0.50** (cheapest, recommended)
- ScraperAPI: ~$0 (within free tier) to $5
- ScrapingBee: ~$0 (within free tier) to $5
- BrightData: ~$3-5

## Time Estimate

With proxy API:
- Per request: ~3-5 seconds
- Total: ~2-3 hours for all 1,924 P-codes
- Much faster than Tor (which would take 12+ days)

## Recommendation

**Use DataImpulse** - At $1/GB, it's the cheapest option and uses residential proxies which are less likely to be blocked. Estimated cost is only ~$0.50 for the full scrape.
