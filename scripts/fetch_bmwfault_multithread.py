#!/usr/bin/env python3
"""
Fetch P-code -> BMW hex mappings from bmwfault.codes with multi-threading support.

Uses DataImpulse proxy for automatic IP rotation.
Multi-threaded for faster scraping (5-10x speedup).
"""

import os
import sys
import json
import sqlite3
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import requests
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Configuration
LOOKUP_URL = "https://bmwfault.codes/Lookup"
REQUEST_DELAY = 0.5  # Small delay between requests (proxy handles rotation)
MAX_WORKERS = 3  # Reduced for CAPTCHA solving
RETRY_ATTEMPTS = 3
BMWFAULT_TURNSTILE_SITEKEY = "0x4AAAAAAAe5OKL31M4ukPJY"

# Thread-local storage for sessions
thread_local = threading.local()


def solve_turnstile(api_key: str) -> Optional[str]:
    """Solve Cloudflare Turnstile CAPTCHA using Capsolver."""
    try:
        resp = requests.post(
            "https://api.capsolver.com/createTask",
            json={
                "clientKey": api_key,
                "task": {
                    "type": "AntiTurnstileTaskProxyLess",
                    "websiteURL": LOOKUP_URL,
                    "websiteKey": BMWFAULT_TURNSTILE_SITEKEY,
                },
            },
            timeout=90,
        )
        
        data = resp.json()
        if data.get("errorId") != 0:
            print(f"    Capsolver error: {data}")
            return None
        
        task_id = data.get("taskId")
        if not task_id:
            return None
        
        # Wait for result
        for _ in range(20):
            time.sleep(3)
            result = requests.post(
                "https://api.capsolver.com/getTaskResult",
                json={"clientKey": api_key, "taskId": task_id},
                timeout=90,
            )
            rj = result.json()
            
            if rj.get("status") == "ready":
                return rj.get("solution", {}).get("token")
            elif rj.get("status") == "failed":
                print(f"    Capsolver task failed: {rj}")
                return None
        
        return None
    except Exception as e:
        print(f"    Capsolver exception: {e}")
        return None

def get_session() -> requests.Session:
    """Get thread-local session with proxy."""
    if not hasattr(thread_local, 'session'):
        thread_local.session = requests.Session()
        
        # Set proxy
        proxy_url = os.environ.get('DATAIMPULSE_PROXY')
        if proxy_url:
            thread_local.session.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
        
        # Set headers
        thread_local.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
        })
    
    return thread_local.session


def get_db_path() -> Path:
    """Get database path."""
    return ROOT / "data" / "databases" / "mist_data.db"


def init_database():
    """Initialize database tables."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    
    # Main table for P-code data
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bmwfault_pcodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pcode TEXT NOT NULL,
            hex_code TEXT NOT NULL,
            label TEXT,
            ecu_variant TEXT,
            ecu_label TEXT,
            fault_info TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pcode, hex_code, ecu_variant)
        )
    """)
    
    # Aggregated mappings
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bmwfault_mappings (
            pcode TEXT PRIMARY KEY,
            hex_codes TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Progress tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bmwfault_fetch_progress (
            pcode TEXT PRIMARY KEY,
            status TEXT,  -- 'success', 'failed', 'not_found'
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def get_missing_pcodes() -> List[str]:
    """Get list of P-codes not yet fetched."""
    # Load forum pcodes
    with open(ROOT / "data" / "forum_pcodes.json") as f:
        all_pcodes = json.load(f)
    
    db_path = get_db_path()
    if not db_path.exists():
        return all_pcodes
    
    conn = sqlite3.connect(str(db_path))
    
    # Get already fetched codes
    try:
        cursor = conn.execute("SELECT pcode FROM bmwfault_fetch_progress WHERE status = 'success'")
        fetched = set(row[0] for row in cursor.fetchall())
    except:
        fetched = set()
    
    conn.close()
    
    missing = [p for p in all_pcodes if p not in fetched]
    return missing


def fetch_pcode_data(pcode: str) -> Dict:
    """Fetch data for a single P-code."""
    session = get_session()
    api_key = os.environ.get('CAPSOLVER_API_KEY', '').strip('"\'')
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Step 1: Get lookup page and extract token
            r = session.get(LOOKUP_URL, timeout=30)
            
            if r.status_code != 200:
                time.sleep(1)
                continue
            
            # Check for Cloudflare challenge
            if 'turnstile' in r.text.lower() or 'cf-turnstile' in r.text.lower():
                if not api_key:
                    return {'pcode': pcode, 'status': 'failed', 'error': 'CAPTCHA required but no API key'}
                
                print(f"    [{pcode}] Solving Turnstile CAPTCHA...")
                turnstile_token = solve_turnstile(api_key)
                if not turnstile_token:
                    return {'pcode': pcode, 'status': 'failed', 'error': 'CAPTCHA solve failed'}
            else:
                turnstile_token = None
            
            # Extract verification token
            token_match = re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', r.text)
            if not token_match:
                return {'pcode': pcode, 'status': 'failed', 'error': 'No token found'}
            
            token = token_match.group(1)
            
            # Step 2: POST search query
            data = {
                'CodePost.textQuery': pcode,
                'CodePost.languageIndex': '2',
                '__RequestVerificationToken': token
            }
            
            if turnstile_token:
                data['cf-turnstile-response'] = turnstile_token
            
            time.sleep(REQUEST_DELAY)  # Small delay
            
            r = session.post(LOOKUP_URL, data=data, timeout=30)
            
            if r.status_code == 429:
                return {'pcode': pcode, 'status': 'rate_limited'}
            
            if r.status_code != 200:
                continue
            
            # Step 3: Parse results
            rows = []
            tbody_match = re.search(r"<tbody>(.*?)</tbody>", r.text, re.DOTALL)
            
            if tbody_match:
                tbody = tbody_match.group(1)
                for tr_match in re.finditer(r"<tr>(.*?)</tr>", tbody, re.DOTALL):
                    tr_content = tr_match.group(1)
                    tds = re.findall(r"<td[^>]*>(.*?)</td>", tr_content, re.DOTALL)
                    
                    if len(tds) >= 6:
                        pcode_val = re.sub(r"<[^>]+>", "", tds[0]).strip()
                        code = re.sub(r"<[^>]+>", "", tds[1]).strip()
                        label = re.sub(r"<[^>]+>", "", tds[2]).strip()
                        ecu_variant = re.sub(r"<[^>]+>", "", tds[3]).strip()
                        ecu_label = re.sub(r"<[^>]+>", "", tds[4]).strip()
                        
                        if pcode_val and code:
                            rows.append({
                                'pcode': pcode_val,
                                'hex_code': code,
                                'label': label,
                                'ecu_variant': ecu_variant,
                                'ecu_label': ecu_label
                            })
            
            if rows:
                return {'pcode': pcode, 'status': 'success', 'rows': rows}
            else:
                return {'pcode': pcode, 'status': 'not_found'}
                
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return {'pcode': pcode, 'status': 'failed', 'error': str(e)}
    
    return {'pcode': pcode, 'status': 'failed', 'error': 'Max retries exceeded'}


def save_result(result: Dict):
    """Save fetch result to database."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    pcode = result['pcode']
    status = result['status']
    
    # Save progress
    conn.execute(
        "INSERT OR REPLACE INTO bmwfault_fetch_progress (pcode, status, fetched_at) VALUES (?, ?, datetime('now'))",
        (pcode, status)
    )
    
    # Save data if successful
    if status == 'success' and 'rows' in result:
        for row in result['rows']:
            conn.execute("""
                INSERT OR REPLACE INTO bmwfault_pcodes 
                (pcode, hex_code, label, ecu_variant, ecu_label)
                VALUES (?, ?, ?, ?, ?)
            """, (row['pcode'], row['hex_code'], row['label'], row['ecu_variant'], row['ecu_label']))
    
    conn.commit()
    conn.close()


def process_pcode(pcode: str) -> Dict:
    """Process a single P-code (wrapper for thread pool)."""
    result = fetch_pcode_data(pcode)
    save_result(result)
    return result


def refresh_mappings():
    """Refresh aggregated mappings table."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    
    # Aggregate
    cursor = conn.execute(
        "SELECT pcode, GROUP_CONCAT(DISTINCT hex_code) FROM bmwfault_pcodes GROUP BY pcode"
    )
    
    for row in cursor:
        pcode, hex_codes = row
        if pcode and hex_codes:
            conn.execute(
                "INSERT OR REPLACE INTO bmwfault_mappings (pcode, hex_codes, updated_at) VALUES (?, ?, datetime('now'))",
                (pcode, hex_codes)
            )
    
    conn.commit()
    conn.close()


def main():
    """Main multi-threaded scraping loop."""
    print("="*60)
    print("Multi-Threaded BMWFault Scraper with DataImpulse")
    print("="*60)
    
    # Check proxy
    proxy = os.environ.get('DATAIMPULSE_PROXY')
    if not proxy:
        print("\nERROR: DATAIMPULSE_PROXY not set!")
        print("Set it with: export DATAIMPULSE_PROXY=http://user:pass@gw.dataimpulse.com:823")
        return 1
    
    print(f"\nUsing proxy: {proxy[:50]}...")
    print(f"Max workers: {MAX_WORKERS}")
    
    # Init database
    init_database()
    
    # Get missing P-codes
    missing = get_missing_pcodes()
    print(f"\nTotal P-codes to fetch: {len(missing)}")
    
    if not missing:
        print("All P-codes already fetched!")
        refresh_mappings()
        return 0
    
    # Process with thread pool
    success_count = 0
    failed_count = 0
    not_found_count = 0
    
    print(f"\nStarting scrape with {MAX_WORKERS} threads...")
    print("-"*60)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_pcode = {
            executor.submit(process_pcode, pcode): pcode 
            for pcode in missing
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_pcode), 1):
            pcode = future_to_pcode[future]
            
            try:
                result = future.result()
                status = result['status']
                
                if status == 'success':
                    success_count += 1
                    row_count = len(result.get('rows', []))
                    hex_codes = [r['hex_code'] for r in result.get('rows', [])]
                    print(f"[{i}/{len(missing)}] {pcode}: ✓ {row_count} rows ({','.join(hex_codes[:3])}{'...' if len(hex_codes) > 3 else ''})")
                elif status == 'not_found':
                    not_found_count += 1
                    print(f"[{i}/{len(missing)}] {pcode}: - Not found")
                else:
                    failed_count += 1
                    error = result.get('error', 'Unknown')
                    print(f"[{i}/{len(missing)}] {pcode}: ✗ Failed ({error[:30]})")
                    
            except Exception as e:
                failed_count += 1
                print(f"[{i}/{len(missing)}] {pcode}: ✗ Exception ({e})")
    
    # Refresh mappings
    refresh_mappings()
    
    # Summary
    print("\n" + "="*60)
    print("SCRAPING COMPLETE")
    print("="*60)
    print(f"Success: {success_count}")
    print(f"Not found: {not_found_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(missing)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
