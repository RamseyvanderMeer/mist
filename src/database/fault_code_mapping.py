"""
OBD-II P-code to BMW ISTA fault code mapping.

BMW ISTA uses hex-style codes (e.g. 2A87, 2A82) while scraped records often
have OBD-II P-codes (e.g. P0015, P0300). This module provides variants to try
when looking up procedures.

Primary source: bmwfault_mappings table (run mist-cli fetch-bmwfault to update)
Fallback: bmwfault_mappings.json, then built-in OBD_TO_BMW
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import List

# Path to scraped bmwfault.codes mapping (fallback when DB unavailable)
_BMWFAULT_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "bmwfault_mappings.json"

# Built-in mappings (fallback when DB and JSON not available)
# Known OBD-II P-code -> BMW ISTA code mappings (from forum/tech sources)
# VANOS/cam timing
OBD_TO_BMW: dict[str, str] = {
    # VANOS / cam timing (bimmerfest, e90post)
    "P0011": "2A81",  # Camshaft Position Timing Over-Advanced Bank 1
    "P0012": "2A82",  # Camshaft Position Timing Over-Retarded Bank 1 (intake)
    "P0015": "2A87",  # Camshaft Position Timing Over-Retarded Bank 2 (exhaust)
    "P0016": "2A88",
    "P0017": "2A89",
    "P0021": "2A8A",
    "P0022": "2A8B",
    # Cylinder-specific misfire (bimmerfest, bimmerforums - 29CC=cyl1 ... 29D2=cyl6)
    "P0301": "29CC",  # Cylinder 1 misfire
    "P0302": "29CD",  # Cylinder 2 misfire
    "P0303": "29CE",  # Cylinder 3 misfire
    "P0304": "29CF",  # Cylinder 4 misfire
    "P0305": "29D0",  # Cylinder 5 misfire
    "P0306": "29D2",  # Cylinder 6 misfire
    # Catalyst efficiency (ISTA often uses stripped hex 420, 430)
    "P0420": "420",   # Catalyst efficiency below threshold Bank 1
    "P0430": "430",   # Catalyst efficiency below threshold Bank 2
    # Fuel trim / oxygen sensor
    "P0171": "171",   # System too lean Bank 1
    "P0174": "174",   # System too lean Bank 2
    "P0135": "135",   # O2 sensor heater Bank 1 Sensor 1
    "P0141": "141",   # O2 sensor heater Bank 1 Sensor 2
    "P0155": "155",   # O2 sensor heater Bank 2 Sensor 1
    "P0161": "161",   # O2 sensor heater Bank 2 Sensor 2
    "P2190": "2190",  # O2 Sensor Signal Stuck Lean Bank 1 Sensor 1
    "P2192": "2192",  # O2 Sensor Signal Stuck Rich Bank 1 Sensor 1
    "P2270": "2270",  # O2 Sensor Stuck Lean Bank 1 Sensor 2
    "P2272": "2272",  # O2 Sensor Stuck Rich Bank 1 Sensor 2
    # Transmission
    "P6800": "6800",  # Transmission pressure control solenoid
    # Evap / purge
    "P0442": "442",   # Evap system leak detected (small)
    "P0455": "455",   # Evap system leak detected (large)
    # MAF / air metering
    "P0101": "101",   # MAF circuit range/performance
    "P0102": "102",   # MAF circuit low input
    "P0103": "103",   # MAF circuit high input
}

# P + 4 hex chars: BMW ISTA often stores without P. E.g. P2A87=2A87, P1632=1632
_P_HEX_PATTERN = re.compile(r"^P([0-9A-Fa-f]{4})$")

# Resolved mapping (bmwfault JSON overrides built-in)
_OBD_TO_BMW_CACHE: dict[str, str] | None = None


def _get_mist_db_path() -> Path:
    """Path to mist_data.db."""
    try:
        from src.database import get_mist_db_path
        return get_mist_db_path()
    except ImportError:
        return Path(__file__).resolve().parent.parent.parent / "data" / "databases" / "mist_data.db"


def _load_from_db() -> dict[str, str]:
    """Load mappings from bmwfault_mappings table."""
    result: dict[str, str] = {}
    db_path = _get_mist_db_path()
    if not db_path.exists():
        return result
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT pcode, hex_codes FROM bmwfault_mappings")
        for row in cur:
            pcode = row["pcode"]
            hex_codes = row["hex_codes"]
            if pcode and hex_codes:
                result[pcode.upper()] = str(hex_codes)
        conn.close()
    except Exception:
        pass
    return result


def _get_obd_to_bmw() -> dict[str, str]:
    """Load mapping: DB > JSON > built-in OBD_TO_BMW."""
    global _OBD_TO_BMW_CACHE
    if _OBD_TO_BMW_CACHE is not None:
        return _OBD_TO_BMW_CACHE
    result = dict(OBD_TO_BMW)
    # 1. DB (primary)
    db_mappings = _load_from_db()
    if db_mappings:
        result.update(db_mappings)
    # 2. JSON fallback
    elif _BMWFAULT_JSON.exists():
        try:
            with open(_BMWFAULT_JSON, encoding="utf-8") as f:
                scraped = json.load(f)
            result.update({k.upper(): str(v) for k, v in scraped.items() if k and v})
        except Exception:
            pass
    _OBD_TO_BMW_CACHE = result
    return result


def get_lookup_variants(code: str) -> List[str]:
    """
    Return fault code variants to try when looking up in ISTA.

    Order: original code, explicit mapping, P2xxx->2xxx strip,
    then stripped without leading zeros (e.g. 0300->300).
    """
    if not code or not isinstance(code, str):
        return []
    code = code.strip().upper()
    variants: List[str] = []

    def _add(v: str) -> None:
        if v and v not in variants:
            variants.append(v)

    # 1. Original
    _add(code)

    # 2. Explicit OBD->BMW mapping (from bmwfault JSON or built-in)
    obd_map = _get_obd_to_bmw()
    if code in obd_map:
        val = obd_map[code]
        # Support comma-separated hexes from bmwfault (e.g. "2862,2A3F,2A69")
        for h in str(val).split(","):
            _add(h.strip())

    # 3. P + 4 hex chars -> strip P (e.g. P2A87->2A87, P0420->0420)
    m = _P_HEX_PATTERN.match(code)
    if m:
        stripped = m.group(1).upper()
        _add(stripped)
        # 4. Without leading zeros (ISTA may store 300 not 0300)
        if stripped.startswith("0") and len(stripped) > 1:
            trimmed = stripped.lstrip("0") or "0"
            _add(trimmed)

    return variants
