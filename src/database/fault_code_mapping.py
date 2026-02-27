"""
OBD-II P-code to BMW ISTA fault code mapping.

BMW ISTA uses hex-style codes (e.g. 2A87, 2A82) while scraped records often
have OBD-II P-codes (e.g. P0015, P0300). This module provides variants to try
when looking up procedures.
"""
import re
from typing import List

# Known OBD-II P-code -> BMW ISTA code mappings (from forum/tech sources)
# Sources: bimmerfest, e90post - VANOS/cam timing codes
# P2190/P6800: O2 sensor / transmission - ISTA uses stripped hex (2190, 6800)
OBD_TO_BMW: dict[str, str] = {
    "P0011": "2A81",  # Camshaft Position Timing Over-Advanced Bank 1
    "P0012": "2A82",  # Camshaft Position Timing Over-Retarded Bank 1 (intake)
    "P0015": "2A87",  # Camshaft Position Timing Over-Retarded Bank 2 (exhaust)
    "P0016": "2A88",
    "P0017": "2A89",
    "P0021": "2A8A",
    "P0022": "2A8B",
    "P2190": "2190",  # O2 Sensor Signal Stuck Lean Bank 1 Sensor 1
    "P6800": "6800",  # Transmission pressure control solenoid
}

# P + 4 hex chars: BMW ISTA often stores without P. E.g. P2A87=2A87, P1632=1632
_P_HEX_PATTERN = re.compile(r"^P([0-9A-Fa-f]{4})$")


def get_lookup_variants(code: str) -> List[str]:
    """
    Return fault code variants to try when looking up in ISTA.

    Order: original code, explicit mapping, P2xxx->2xxx strip.
    """
    if not code or not isinstance(code, str):
        return []
    code = code.strip().upper()
    variants: List[str] = []

    # 1. Original
    variants.append(code)

    # 2. Explicit OBD->BMW mapping
    if code in OBD_TO_BMW:
        bmw = OBD_TO_BMW[code]
        if bmw not in variants:
            variants.append(bmw)

    # 3. P + 4 hex chars -> strip P (e.g. P2A87->2A87, P1632->1632)
    m = _P_HEX_PATTERN.match(code)
    if m:
        stripped = m.group(1).upper()
        if stripped not in variants:
            variants.append(stripped)

    return variants
