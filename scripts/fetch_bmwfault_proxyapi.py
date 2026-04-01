#!/usr/bin/env python3
"""
BMWFault.codes scraper with proxy API (ScraperAPI/ScrapingBee).

This version uses a proxy API service for automatic IP rotation,
avoiding rate limits and blocks.
"""

import os
import sys
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configuration
REQUEST_DELAY_SEC = 2  # Shorter delay since proxy rotates IPs
BATCH_SIZE = 10
LOOKUP_URL = "https://bmwfault.codes/Lookup"
DIAGVIEW_BASE = "https://bmwfault.codes/DiagView"

def get_proxy_config() -> Dict[str, str]:
    """Get proxy configuration from environment."""
    # Try DataImpulse first (recommended)
    datadimpulse_proxy = os.environ.get('DATAIMPULSE_PROXY', '').strip()
    if datadimpulse_proxy:
        return {
            'type': 'dataimpulse',
            'proxy_url': datadimpulse_proxy,
            'headers': {}
        }
    
    # Try ScraperAPI
    scraperapi_key = os.environ.get('SCRAPERAPI_KEY', '').strip()
    if scraperapi_key:
        return {
            'type': 'scraperapi',
            'proxy_url': f'http://api.scraperapi.com?api_key={scraperapi_key}&url=',
            'headers': {}
        }
    
    # Try ScrapingBee
    scrapingbee_key = os.environ.get('SCRAPINGBEE_KEY', '').strip()
    if scrapingbee_key:
        return {
            'type': 'scrapingbee',
            'proxy_url': None,  # Uses different mechanism
            'api_key': scrapingbee_key,
            'headers': {}
        }
    
    # Try BrightData
    brightdata_proxy = os.environ.get('BRIGHTDATA_PROXY', '').strip()
    if brightdata_proxy:
        return {
            'type': 'brightdata',
            'proxy_url': brightdata_proxy,
            'headers': {}
        }
    
    return None

def make_request(url: str, proxy_config: Dict, method='GET', data=None, timeout=30) -> requests.Response:
    """Make request through proxy API."""
    proxy_type = proxy_config.get('type')
    
    if proxy_type == 'dataimpulse':
        # DataImpulse: use as regular HTTP proxy with auth
        proxies = {
            'http': proxy_config['proxy_url'],
            'https': proxy_config['proxy_url']
        }
        return requests.request(method, url, data=data, proxies=proxies, timeout=timeout)
    
    elif proxy_type == 'scraperapi':
        # ScraperAPI: prepend URL with proxy endpoint
        full_url = f"{proxy_config['proxy_url']}{requests.utils.quote(url, safe='')}&premium=true&country_code=us"
        return requests.request(method, full_url, timeout=timeout)
    
    elif proxy_type == 'scrapingbee':
        # ScrapingBee: use their API endpoint
        params = {
            'api_key': proxy_config['api_key'],
            'url': url,
            'premium_proxy': 'true',
            'country_code': 'us'
        }
        if method == 'POST' and data:
            params['js_scenario'] = json.dumps([{"instruction": "fill_form", "params": data}])
        
        api_url = "https://app.scrapingbee.com/api/v1/"
        return requests.get(api_url, params=params, timeout=timeout)
    
    elif proxy_type == 'brightdata':
        # BrightData: use as regular proxy
        proxies = {
            'http': proxy_config['proxy_url'],
            'https': proxy_config['proxy_url']
        }
        return requests.request(method, url, data=data, proxies=proxies, timeout=timeout)
    
    else:
        # No proxy - direct request
        return requests.request(method, url, data=data, timeout=timeout)

def fetch_pcode_mapping(pcode: str, proxy_config: Dict) -> List[Dict]:
    """Fetch mapping for a single P-code."""
    try:
        # Step 1: Get the lookup page
        r = make_request(LOOKUP_URL, proxy_config)
        
        if r.status_code != 200:
            print(f"  [{pcode}] HTTP {r.status_code}")
            return []
        
        # Extract verification token
        import re
        token_match = re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', r.text)
        if not token_match:
            print(f"  [{pcode}] No token found")
            return []
        
        token = token_match.group(1)
        
        # Step 2: POST search query
        data = {
            'CodePost.textQuery': pcode,
            'CodePost.languageIndex': '2',
            '__RequestVerificationToken': token
        }
        
        r = make_request(LOOKUP_URL, proxy_config, method='POST', data=data)
        
        if r.status_code != 200:
            print(f"  [{pcode}] POST failed: {r.status_code}")
            return []
        
        # Parse results
        rows = []
        tbody_match = re.search(r"<tbody>(.*?)</tbody>", r.text, re.DOTALL)
        if tbody_match:
            tbody = tbody_match.group(1)
            for tr_match in re.finditer(r"<tr>(.*?)</tr>", tbody, re.DOTALL):
                tr_content = tr_match.group(1)
                tds = re.findall(r"<td[^>]*>(.*?)</td>", tr_content, re.DOTALL)
                if len(tds) >= 6:
                    import html
                    pcode_val = html.unescape(re.sub(r"<[^>]+>", "", tds[0])).strip()
                    code = html.unescape(re.sub(r"<[^>]+>", "", tds[1])).strip()
                    label = html.unescape(re.sub(r"<[^>]+>", "", tds[2])).strip()
                    ecu_variant = html.unescape(re.sub(r"<[^>]+>", "", tds[3])).strip()
                    ecu_label = html.unescape(re.sub(r"<[^>]+>", "", tds[4])).strip()
                    
                    if pcode_val and code:
                        rows.append({
                            'pcode': pcode_val,
                            'hex_code': code,
                            'label': label,
                            'ecu_variant': ecu_variant,
                            'ecu_label': ecu_label
                        })
        
        return rows
        
    except Exception as e:
        print(f"  [{pcode}] Error: {e}")
        return []

def save_to_db(rows: List[Dict]):
    """Save fetched rows to database."""
    if not rows:
        return
    
    db_path = ROOT / "data" / "databases" / "mist_data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    
    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bmwfault_pcodes_proxy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pcode TEXT NOT NULL,
            hex_code TEXT NOT NULL,
            label TEXT,
            ecu_variant TEXT,
            ecu_label TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pcode, hex_code, ecu_variant)
        )
    """)
    
    for row in rows:
        conn.execute("""
            INSERT OR REPLACE INTO bmwfault_pcodes_proxy 
            (pcode, hex_code, label, ecu_variant, ecu_label)
            VALUES (?, ?, ?, ?, ?)
        """, (row['pcode'], row['hex_code'], row['label'], row['ecu_variant'], row['ecu_label']))
    
    conn.commit()
    conn.close()

def get_missing_pcodes() -> List[str]:
    """Get list of P-codes we don't have yet."""
    # Load forum pcodes
    with open(ROOT / "data" / "forum_pcodes.json") as f:
        all_pcodes = json.load(f)
    
    # Check what we have (from both old and new tables)
    db_path = ROOT / "data" / "databases" / "mist_data.db"
    have = set()
    
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        # Check new proxy table
        try:
            cursor = conn.execute("SELECT DISTINCT pcode FROM bmwfault_pcodes_proxy")
            have.update(row[0] for row in cursor.fetchall())
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet
        
        # Also check original table
        try:
            cursor = conn.execute("SELECT DISTINCT pcode FROM bmwfault_pcodes")
            have.update(row[0] for row in cursor.fetchall())
        except sqlite3.OperationalError:
            pass
        
        conn.close()
    
    missing = [p for p in all_pcodes if p not in have]
    return missing

def main():
    """Main scraping loop."""
    print("="*60)
    print("BMWFault.codes Scraper with Proxy API")
    print("="*60)
    
    # Check proxy config
    proxy_config = get_proxy_config()
    if not proxy_config:
        print("\nERROR: No proxy API configured!")
        print("Set one of these environment variables:")
        print("  - SCRAPERAPI_KEY")
        print("  - SCRAPINGBEE_KEY")
        print("  - BRIGHTDATA_PROXY")
        return 1
    
    print(f"\nUsing proxy: {proxy_config['type']}")
    
    # Get missing P-codes
    missing = get_missing_pcodes()
    print(f"\nTotal P-codes to fetch: {len(missing)}")
    
    if not missing:
        print("All P-codes already fetched!")
        return 0
    
    # Fetch each P-code
    total_fetched = 0
    for i, pcode in enumerate(missing, 1):
        print(f"\n[{i}/{len(missing)}] Fetching {pcode}...")
        
        rows = fetch_pcode_mapping(pcode, proxy_config)
        
        if rows:
            save_to_db(rows)
            hex_codes = [r['hex_code'] for r in rows]
            print(f"  ✓ Saved {len(rows)} rows: {','.join(hex_codes[:5])}{'...' if len(hex_codes) > 5 else ''}")
            total_fetched += len(rows)
        else:
            print(f"  ✗ No results")
        
        # Delay between requests
        if i < len(missing):
            time.sleep(REQUEST_DELAY_SEC)
    
    print(f"\n{'='*60}")
    print(f"Complete! Fetched {total_fetched} total rows")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
