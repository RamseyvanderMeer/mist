#!/usr/bin/env python3
"""
Fetch P-code -> BMW hex mappings and full row data from bmwfault.codes.

Stores:
  - bmwfault_pcodes: full rows (PCode, Code, Label, ECU Variant, ECU Label, fault_info JSON)
  - bmwfault_mappings: aggregated pcode -> hex_codes for get_lookup_variants()

HTTP mode (default):
  - Uses requests with cookies from .env (CLOUDFLARE_COOKIES) or data/bmwfault_cookies.txt
  - CAPSOLVER_API_KEY for Turnstile. Cookie refresh via --refresh-cookies (HTTP+Capsolver).
  - BMWFAULT_PROXY (or HTTP_PROXY) for proxy when IP is blocked ("no longer available to you").
"""
from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

BATCH_SIZE = 10
REQUEST_DELAY_SEC = 5  # delay between P-codes
DIAGVIEW_DELAY_SEC = 3  # delay between DiagView fetches
RATE_LIMIT_WAIT_BASE_SEC = 540   # 9 min
RATE_LIMIT_BACKOFF_FACTOR = 3    # 9 -> 27 -> 81 min
BLOCKED_WAIT_SEC = 600           # 10 min for IP block (matches VPN IP rotation ~5–10 min)

LOOKUP_URL = "https://bmwfault.codes/Lookup"
DIAGVIEW_BASE = "https://bmwfault.codes/DiagView"
BMWFAULT_TURNSTILE_SITEKEY = "0x4AAAAAAAe5OKL31M4ukPJY"

_VERBOSE = False


def _v(msg: str) -> None:
    if _VERBOSE:
        print(f"  [v] {msg}")


def _get_mist_db_path() -> Path:
    try:
        from src.database import get_mist_db_path
        return get_mist_db_path()
    except ImportError:
        return ROOT / "data" / "databases" / "mist_data.db"


def _load_checkpoint() -> str | None:
    db_path = _get_mist_db_path()
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT last_pcode FROM bmwfault_fetch_checkpoint WHERE id = 1"
        ).fetchone()
        conn.close()
        if row and row[0]:
            return str(row[0]).strip().upper()
    except Exception:
        pass
    return None


def _save_checkpoint(last_pcode: str) -> None:
    db_path = _get_mist_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bmwfault_fetch_checkpoint (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            last_pcode TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO bmwfault_fetch_checkpoint (id, last_pcode, updated_at) VALUES (1, ?, datetime('now'))",
        (last_pcode.upper(),),
    )
    conn.commit()
    conn.close()


def _get_cookies() -> str | None:
    env_val = os.environ.get("CLOUDFLARE_COOKIES", "").strip()
    if env_val:
        return env_val
    cookies_file = ROOT / "data" / "bmwfault_cookies.txt"
    if cookies_file.exists():
        return cookies_file.read_text(encoding="utf-8").strip()
    return None


def _get_proxies() -> dict[str, str] | None:
    """Get proxy config from BMWFAULT_PROXY, or HTTP_PROXY/HTTPS_PROXY. Returns dict for requests."""
    proxy = (
        os.environ.get("BMWFAULT_PROXY")
        or os.environ.get("bmwfault_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("HTTPS_PROXY")
    )
    if not proxy or not proxy.strip():
        return None
    proxy = proxy.strip()
    if not proxy.startswith(("http://", "https://", "socks4://", "socks5://")):
        proxy = "http://" + proxy
    return {"http": proxy, "https": proxy}


def _is_rate_limited(html_text: str) -> bool:
    """Check if response indicates rate limit (app or Cloudflare)."""
    lower = html_text.lower()
    return (
        "too often" in lower
        or "try again later" in lower
        or "performing that action" in lower
        or "rate limit" in lower
        or "being rate limited" in lower
        or "complete the captcha verification" in lower
    )


def _is_blocked(html_text: str) -> bool:
    """Check if response indicates IP/service blocked (e.g. 'no longer available to you')."""
    lower = html_text.lower()
    return "no longer available to you" in lower or "service is not available" in lower


def _extract_verification_token(html_text: str) -> str | None:
    m = re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', html_text)
    if m:
        return m.group(1)
    m = re.search(r'value="([^"]+)"[^>]*name="__RequestVerificationToken"', html_text)
    if m:
        return m.group(1)
    return None


def _parse_diagview_fault_info(html_text: str) -> dict[str, str]:
    """Parse DiagView page into JSON. Sections: fault_code_description, fault_code_conditions, fault_time_condition, service_plan, fault_impact, warnings, service_notes."""
    result: dict[str, str] = {}
    sections = [
        ("Fault Code Description", "fault_code_description"),
        ("Fault Code Conditions", "fault_code_conditions"),
        ("Fault Time Condition", "fault_time_condition"),
        ("Service Plan", "service_plan"),
        ("Fault Impact", "fault_impact"),
        ("Warnings", "warnings"),
        ("Service Notes", "service_notes"),
    ]
    for header, key in sections:
        # Match: <thead class="thead-dark"><tr><th ...>Header</th></tr></thead><tbody>...</tbody>
        pat = rf'<thead[^>]*>.*?{re.escape(header)}.*?</thead>\s*<tbody>(.*?)</tbody>'
        m = re.search(pat, html_text, re.DOTALL | re.IGNORECASE)
        if m:
            tbody = m.group(1)
            # Extract text from td cells, strip tags, decode entities
            parts = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tbody, re.DOTALL)
            text_parts = []
            for p in parts:
                cleaned = re.sub(r"<[^>]+>", " ", p)
                cleaned = html.unescape(cleaned).strip()
                if cleaned:
                    text_parts.append(cleaned)
            result[key] = "\n".join(text_parts).strip()
        else:
            result[key] = ""
    return result


def _extract_diagview_id_from_link(fault_info_cell: str) -> str | None:
    """Extract id from <a href="/DiagView?id=XXX">View</a>."""
    m = re.search(r'href="/DiagView\?id=([^"&]+)"', fault_info_cell, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _parse_lookup_rows(html_text: str) -> list[tuple[str, str, str, str, str, str]]:
    """Parse Lookup results table. Returns [(pcode, code, label, ecu_variant, ecu_label, fault_info_cell), ...]."""
    rows = []
    # Find tbody
    m = re.search(r"<tbody>(.*?)</tbody>", html_text, re.DOTALL)
    if not m:
        return rows
    tbody = m.group(1)
    for tr_match in re.finditer(r"<tr>(.*?)</tr>", tbody, re.DOTALL):
        tr_content = tr_match.group(1)
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr_content, re.DOTALL)
        if len(tds) >= 6:
            pcode = html.unescape(re.sub(r"<[^>]+>", "", tds[0])).strip()
            code = html.unescape(re.sub(r"<[^>]+>", "", tds[1])).strip()
            label = html.unescape(re.sub(r"<[^>]+>", "", tds[2])).strip()
            ecu_variant = html.unescape(re.sub(r"<[^>]+>", "", tds[3])).strip()
            ecu_label = html.unescape(re.sub(r"<[^>]+>", "", tds[4])).strip()
            fault_info_cell = tds[5]
            if pcode and code:
                rows.append((pcode, code, label, ecu_variant, ecu_label, fault_info_cell))
    return rows


def _parse_hex_codes_from_rows(rows: list[tuple[str, str, str, str, str, str]]) -> list[str]:
    """Unique hex codes from rows (for bmwfault_mappings aggregate)."""
    seen: set[str] = set()
    for _, code, _, _, _, _ in rows:
        if code and code not in seen:
            seen.add(code)
    return sorted(seen)


def _fetch_diagview(session, diag_view_id: str) -> dict[str, str] | None:
    """Fetch DiagView page and return parsed fault info JSON."""
    url = f"{DIAGVIEW_BASE}?id={diag_view_id}"
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        info = _parse_diagview_fault_info(r.text)
        # Debug: save first empty DiagView HTML for inspection (challenge page or different structure)
        if info and not any(v for v in info.values() if v):
            debug_path = ROOT / "data" / "bmwfault_debug_diagview_empty.html"
            if not debug_path.exists():
                debug_path.write_text(r.text, encoding="utf-8")
                print(f"  [debug] Saved empty DiagView sample to {debug_path}")
        return info
    except Exception:
        return None


def _save_pcodes_to_db(rows: list[tuple[str, str, str, str, str, str | None]]) -> None:
    """Insert/replace rows into bmwfault_pcodes."""
    if not rows:
        return
    try:
        from src.database import ensure_mist_database
        ensure_mist_database()
    except ImportError:
        pass
    db_path = _get_mist_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bmwfault_pcodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pcode TEXT NOT NULL,
            code TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT '',
            ecu_variant TEXT NOT NULL DEFAULT '',
            ecu_label TEXT NOT NULL DEFAULT '',
            fault_info TEXT,
            diag_view_id TEXT NOT NULL DEFAULT '',
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pcode, code, ecu_variant, diag_view_id)
        )
        """
    )
    for pcode, code, label, ecu_variant, ecu_label, fault_info_json in rows:
        diag_id = ""
        to_store = fault_info_json
        if fault_info_json:
            try:
                j = json.loads(fault_info_json) if isinstance(fault_info_json, str) else fault_info_json
                diag_id = str(j.pop("_diag_view_id", ""))
                to_store = json.dumps(j, ensure_ascii=False) if j else None
            except Exception:
                to_store = fault_info_json
        conn.execute(
            """
            INSERT OR REPLACE INTO bmwfault_pcodes (pcode, code, label, ecu_variant, ecu_label, fault_info, diag_view_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (pcode, code, label, ecu_variant, ecu_label, to_store, diag_id),
        )
    conn.commit()
    conn.close()


def _refresh_bmwfault_mappings_from_pcodes() -> None:
    """Aggregate bmwfault_pcodes -> bmwfault_mappings (pcode -> comma-separated hex codes)."""
    db_path = _get_mist_db_path()
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bmwfault_mappings (
            pcode TEXT PRIMARY KEY,
            hex_codes TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur = conn.execute(
        "SELECT pcode, GROUP_CONCAT(DISTINCT code) FROM bmwfault_pcodes GROUP BY pcode"
    )
    for row in cur:
        pcode, hex_codes = row[0], row[1]
        if pcode and hex_codes:
            conn.execute(
                "INSERT OR REPLACE INTO bmwfault_mappings (pcode, hex_codes, updated_at) VALUES (?, ?, datetime('now'))",
                (pcode.upper(), hex_codes),
            )
    conn.commit()
    conn.close()


def _run_http_for_pcode(session, pcode: str, token: str, api_key: str, fetch_diagview: bool) -> tuple[list[tuple[str, str, str, str, str, str | None]], str]:
    """Fetch one P-code via HTTP, parse rows, optionally fetch DiagView for View links."""
    import requests
    data = {
        "CodePost.textQuery": pcode,
        "CodePost.languageIndex": "2",
        "__RequestVerificationToken": token,
    }
    if api_key:
        turnstile = _solve_turnstile(api_key)
        if turnstile:
            data["cf-turnstile-response"] = turnstile
        else:
            # Don't POST without token - triggers rate limit or captcha; signal retry
            raise RuntimeError("Turnstile solve failed (Capsolver timeout or error); will retry after backoff")
    r = session.post(LOOKUP_URL, data=data, timeout=20)
    r.raise_for_status()
    rows_raw = _parse_lookup_rows(r.text)
    result = []
    for pcode_val, code, label, ecu_variant, ecu_label, fault_info_cell in rows_raw:
        diag_id = _extract_diagview_id_from_link(fault_info_cell)
        fault_info_json = None
        if fetch_diagview and diag_id:
            time.sleep(DIAGVIEW_DELAY_SEC)
            info = _fetch_diagview(session, diag_id)
            if info:
                info["_diag_view_id"] = diag_id
                fault_info_json = json.dumps(info, ensure_ascii=False)
        result.append((pcode_val, code, label, ecu_variant, ecu_label, fault_info_json))
    return (result, r.text)


def _solve_turnstile(api_key: str) -> str | None:
    try:
        import requests
    except ImportError:
        return None
    capsolver_timeout = 90  # Capsolver can be slow; increase for VPN
    for create_attempt in range(3):
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
                timeout=capsolver_timeout,
            )
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if create_attempt < 2:
                print(f"Capsolver API timeout/connection error, retrying in 10s... ({e})")
                time.sleep(10)
                continue
            print(f"Capsolver createTask failed after retries: {e}")
            return None
    data = resp.json()
    if data.get("errorId") != 0:
        print(f"Capsolver error: {data}")
        return None
    task_id = data.get("taskId")
    if not task_id:
        return None
    for _ in range(18):
        time.sleep(2)
        try:
            result = requests.post(
                "https://api.capsolver.com/getTaskResult",
                json={"clientKey": api_key, "taskId": task_id},
                timeout=capsolver_timeout,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if _ < 17:
                continue
            print(f"Capsolver getTaskResult failed: {e}")
            return None
        rj = result.json()
        if rj.get("status") == "ready":
            return rj.get("solution", {}).get("token")
        if rj.get("status") == "failed":
            print(f"Capsolver task failed: {rj}")
            return None
    return None


def _refresh_cookies_via_http() -> bool:
    try:
        import requests
    except ImportError:
        return False
    api_key = os.environ.get("CAPSOLVER_API_KEY") or os.environ.get("capsolver_api_key", "").strip('"\'')
    if not api_key:
        return False
    cookie_file = ROOT / "data" / "bmwfault_cookies.txt"
    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/145.0.0.0 Safari/537.36",
        "referer": "https://bmwfault.codes/Lookup",
    })
    proxies = _get_proxies()
    if proxies:
        session.proxies.update(proxies)
        if _VERBOSE:
            _v(f"Using proxy: {proxies.get('https', proxies.get('http', '?'))[:50]}...")
    cookies_str = _get_cookies()
    if cookies_str:
        for part in cookies_str.split(";"):
            part = part.strip()
            if "=" in part:
                name, _, value = part.partition("=")
                session.cookies.set(name.strip(), value.strip(), domain=".bmwfault.codes")
    try:
        r = session.get(LOOKUP_URL, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"HTTP refresh failed: {e}")
        return False
    token = _extract_verification_token(r.text)
    if not token:
        return False
    turnstile = _solve_turnstile(api_key)
    if not turnstile:
        return False
    data = {
        "CodePost.textQuery": "P0000",
        "CodePost.languageIndex": "2",
        "__RequestVerificationToken": token,
        "cf-turnstile-response": turnstile,
    }
    try:
        r = session.post(LOOKUP_URL, data=data, timeout=20)
        r.raise_for_status()
    except Exception:
        return False
    vals = [f"{c.name}={c.value}" for c in session.cookies]
    if vals:
        cookie_file.write_text("; ".join(vals), encoding="utf-8")
        print(f"Saved cookies to {cookie_file} (via HTTP+Capsolver)")
        return True
    return False


def _load_codes_from_db(limit: int | None = None, verbose: bool = False) -> list[str]:
    """Load P-codes from ISTA DB (if it has P-codes) or forum_config fallback."""
    try:
        from src.paths import get_paths
        paths = get_paths()
        env_path = os.environ.get("ISTA_DB_PATH")
        db_path = Path(env_path).expanduser().resolve() if env_path else paths.get_database_path("DiagDocDb_Decrypted.sqlite")
        if not db_path.exists():
            db_path = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
        if db_path.exists():
            from src.database.ista_db import IstaDatabase
            db = IstaDatabase(db_path=str(db_path))
            codes = db.get_all_fault_codes(code_pattern="P%", limit=limit)
            db.close()
            if codes:
                if verbose:
                    print(f"[ISTA DB] Loaded {len(codes)} P-codes")
                return codes
            if verbose:
                print("[ISTA DB] No P-codes (stores hex format); using forum_config")
    except Exception as e:
        if verbose:
            print(f"[ISTA DB] {e}; using forum_config")
    try:
        from scrapers.utils.forum_config import FAULT_CODES_TO_SEARCH
        pcodes = [c for c in FAULT_CODES_TO_SEARCH if isinstance(c, str) and c.upper().startswith("P")]
        if pcodes:
            return pcodes[:limit] if limit else pcodes
    except Exception:
        pass
    fallback = [
        "P0011", "P0012", "P0015", "P0016", "P0017", "P0021", "P0022",
        "P0101", "P0102", "P0103", "P0135", "P0141", "P0155", "P0161",
        "P0171", "P0174", "P0300", "P0301", "P0302", "P0303", "P0304", "P0305", "P0306",
        "P0420", "P0430", "P0442", "P0455", "P1055", "P0000", "P0001", "P0002", "P0003", "P0004",
    ]
    return fallback[:limit] if limit else fallback


def run(
    codes: list[str] | None = None,
    limit: int | None = None,
    http_only: bool = True,
    resume: bool = True,
    verbose: bool = False,
    fetch_diagview: bool = True,
) -> int:
    global _VERBOSE
    _VERBOSE = verbose
    if codes is None:
        codes = _load_codes_from_db(limit=limit, verbose=_VERBOSE)
    if limit:
        codes = codes[:limit]
    if _VERBOSE and codes:
        print(f"[Fetch] {len(codes)} P-codes to process")
    cookies_str = _get_cookies()
    if not cookies_str:
        print("Set CLOUDFLARE_COOKIES in .env or create data/bmwfault_cookies.txt")
        return 1
    if not resume:
        conn = sqlite3.connect(str(_get_mist_db_path()))
        conn.execute("DELETE FROM bmwfault_fetch_checkpoint WHERE id = 1")
        conn.commit()
        conn.close()
    api_key = os.environ.get("CAPSOLVER_API_KEY") or os.environ.get("capsolver_api_key", "").strip('"\'')
    last_pcode = _load_checkpoint()
    skip_until_passed = bool(last_pcode)
    if last_pcode:
        print(f"Resuming from checkpoint (last: {last_pcode})")
    try:
        import requests
    except ImportError:
        print("pip install requests")
        return 1
    session = requests.Session()
    session.headers.update({
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/145.0.0.0 Safari/537.36",
        "referer": "https://bmwfault.codes/Lookup",
    })
    proxies = _get_proxies()
    if proxies:
        session.proxies.update(proxies)
        if _VERBOSE:
            _v(f"Using proxy: {proxies.get('https', proxies.get('http', '?'))[:50]}...")
    for part in cookies_str.split(";"):
        part = part.strip()
        if "=" in part:
            name, _, value = part.partition("=")
            session.cookies.set(name.strip(), value.strip(), domain=".bmwfault.codes")
    r = session.get(LOOKUP_URL, timeout=15)
    r.raise_for_status()
    token = _extract_verification_token(r.text)
    if not token:
        print("No verification token - try --refresh-cookies")
        return 1
    saved = 0
    for i, pcode in enumerate(codes):
        pcode = pcode.strip().upper()
        if not pcode.startswith("P"):
            continue
        if skip_until_passed:
            if pcode == last_pcode:
                skip_until_passed = False
            continue
        attempt = 0
        while True:
            try:
                rows, r_text = _run_http_for_pcode(session, pcode, token, api_key, fetch_diagview=fetch_diagview)
                if _is_rate_limited(r_text):
                    wait_sec = RATE_LIMIT_WAIT_BASE_SEC * (RATE_LIMIT_BACKOFF_FACTOR ** attempt)
                    print(f"[{pcode}] Rate limited, waiting {wait_sec // 60} min (attempt {attempt + 1})...")
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                if _is_blocked(r_text):
                    wait_sec = BLOCKED_WAIT_SEC  # fixed 10 min for VPN IP rotation
                    print(f"[{pcode}] IP blocked, waiting {wait_sec // 60} min for VPN rotation (attempt {attempt + 1})...")
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                break
            except RuntimeError as e:
                if "Turnstile solve failed" in str(e):
                    wait_sec = RATE_LIMIT_WAIT_BASE_SEC * (RATE_LIMIT_BACKOFF_FACTOR ** attempt)
                    print(f"[{pcode}] Capsolver failed, waiting {wait_sec // 60} min before retry (attempt {attempt + 1})...")
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait_sec = RATE_LIMIT_WAIT_BASE_SEC * (RATE_LIMIT_BACKOFF_FACTOR ** attempt)
                    print(f"[{pcode}] HTTP 429 rate limited, waiting {wait_sec // 60} min (attempt {attempt + 1})...")
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                print(f"[{pcode}] HTTP Error: {e}")
                rows, r_text = [], ""
                break
            except Exception as e:
                print(f"[{pcode}] Error: {e}")
                rows, r_text = [], ""
                break
        next_token = _extract_verification_token(r_text)
        if next_token:
            token = next_token
        if rows:
            hexes = _parse_hex_codes_from_rows([
                (r[0], r[1], r[2], r[3], r[4], "") for r in rows
            ])
            print(f"  {pcode} -> {','.join(hexes)} ({len(rows)} rows)")
            _save_pcodes_to_db(rows)
            saved += len(rows)
        else:
            # Debug: if page has error alert, save for inspection (possible rate limit missed)
            if r_text and "alert-danger" in r_text and _VERBOSE:
                debug_path = ROOT / "data" / "bmwfault_debug_empty_with_alert.html"
                debug_path.write_text(r_text, encoding="utf-8")
                print(f"  [debug] Saved empty response with alert to {debug_path}")
            print(f"  [{pcode}] No results")
        _save_checkpoint(pcode)
        if not token:
            r = session.get(LOOKUP_URL, timeout=15)
            r.raise_for_status()
            token = _extract_verification_token(r.text)
        if i < len(codes) - 1:
            time.sleep(REQUEST_DELAY_SEC)
    _refresh_bmwfault_mappings_from_pcodes()
    if saved:
        print(f"\nSaved {saved} rows to bmwfault_pcodes, refreshed bmwfault_mappings")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Fetch P->hex mappings and full row data from bmwfault.codes")
    ap.add_argument("--codes", type=str, help="Comma-separated P-codes")
    ap.add_argument("--limit", type=int, help="Limit number of codes")
    ap.add_argument("--no-diagview", action="store_true", help="Skip DiagView fetches (fastest)")
    ap.add_argument("--refresh-cookies", action="store_true", help="Refresh cookies via HTTP+Capsolver")
    ap.add_argument("--no-resume", action="store_true", help="Start fresh")
    ap.add_argument("--proxy", type=str, help="Proxy URL (e.g. http://user:pass@host:port) for blocked IPs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.proxy:
        os.environ["BMWFAULT_PROXY"] = args.proxy
    if args.refresh_cookies:
        _refresh_cookies_via_http()
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else None
    return run(
        codes=codes,
        limit=args.limit,
        resume=not args.no_resume,
        verbose=args.verbose,
        fetch_diagview=not args.no_diagview,
    )


if __name__ == "__main__":
    sys.exit(main())
