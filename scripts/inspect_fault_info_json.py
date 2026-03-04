#!/usr/bin/env python3
"""
Inspect fault_info JSON column in bmwfault_pcodes.
Check if DiagView page extraction produced any content.
"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
db = ROOT / "data" / "databases" / "mist_data.db"
if not db.exists():
    print("DB not found at", db)
    sys.exit(1)

conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row

# Rows with fault_info
cur = conn.execute("""
    SELECT pcode, code, ecu_variant, fault_info, diag_view_id
    FROM bmwfault_pcodes
    WHERE fault_info IS NOT NULL AND fault_info != ''
    ORDER BY pcode, code
    LIMIT 20
""")
rows = cur.fetchall()
print(f"Rows with fault_info: checking first {len(rows)}")
print()

# Stats: how many have ANY non-empty field?
has_content = 0
samples_with_content = []
for r in rows:
    try:
        j = json.loads(r["fault_info"])
        content = {k: v for k, v in j.items() if not k.startswith("_") and v}
        if content:
            has_content += 1
            if len(samples_with_content) < 3:
                samples_with_content.append((r, j))
    except json.JSONDecodeError:
        pass

print(f"Rows with non-empty fault_info content: {has_content}")
print()

if samples_with_content:
    print("Sample fault_info with content:")
    for r, j in samples_with_content[:1]:  # Full JSON for first sample
        print(f"\n  {r['pcode']} | {r['code']} | {r['ecu_variant']}")
        print("  Full fault_info JSON:")
        clean = {k: v for k, v in j.items() if not k.startswith("_")}
        print(json.dumps(clean, indent=4, ensure_ascii=False)[:1500])
        if len(json.dumps(clean)) > 1500:
            print("  ... (truncated)")
else:
    print("No fault_info rows have content - all are empty.")
    print("\nRaw fault_info sample (first row):")
    if rows:
        r = rows[0]
        print(f"  pcode={r['pcode']} code={r['code']} ecu={r['ecu_variant']}")
        print(f"  diag_view_id={r['diag_view_id']}")
        print(f"  fault_info (first 500 chars):")
        print(f"    {repr(r['fault_info'][:500])}")

# P0016 specifically
cur2 = conn.execute("""
    SELECT code, ecu_variant, fault_info FROM bmwfault_pcodes
    WHERE pcode = 'P0016' AND fault_info IS NOT NULL AND fault_info != ''
    LIMIT 3
""")
p0016_rows = cur2.fetchall()
if p0016_rows:
    has_any = False
    for r in p0016_rows:
        try:
            j = json.loads(r[2])
            if any(v for k, v in j.items() if not k.startswith("_") and v):
                has_any = True
                break
        except Exception:
            pass
    print("\n--- P0016 fault_info ---")
    print(f"P0016 rows with fault_info: {len(p0016_rows)} sampled")
    print(f"Has non-empty content: {'YES' if has_any else 'NO (DiagView returned empty/challenge pages)'}")

conn.close()
