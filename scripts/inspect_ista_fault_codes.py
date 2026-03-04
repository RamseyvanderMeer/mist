#!/usr/bin/env python3
"""
Inspect BMW ISTA fault code tables to discover what codes exist and potential mappings.

Run when DiagDocDb_Decrypted.sqlite is available. Outputs:
- Schema of fault-related tables (XEP_FAULTCODES, XEP_FAULTLABELS, XEP_COMBINEDFAULTS)
- Sample and distinct CODE values from XEP_FAULTCODES
- Optional: export to data/fault_code_inventory.json for use by fault_code_mapping

Usage:
  python scripts/inspect_ista_fault_codes.py
  python scripts/inspect_ista_fault_codes.py --export  # also write JSON inventory
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.paths import get_paths
from sqlalchemy import create_engine, text, inspect


def main():
    paths = get_paths()
    for name in ("DiagDocDb_Decrypted.sqlite", "DiagDocDb_DECRYPTED.sqlite"):
        db_path = paths.get_database_path(name)
        if db_path.exists():
            break
    else:
        print("ISTA database not found. Expected DiagDocDb_Decrypted.sqlite in data/databases/")
        return 1

    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    # 1. List fault-related tables
    fault_tables = [t for t in inspector.get_table_names() 
                    if "FAULT" in t.upper() or "XEP_" in t and "FAULT" in t.upper()]
    print("=" * 60)
    print("Fault-related tables:")
    for t in sorted(fault_tables):
        print(f"  {t}")

    # 2. Schema of key tables
    for table in ["XEP_FAULTCODES", "XEP_FAULTLABELS", "XEP_COMBINEDFAULTS"]:
        if table not in inspector.get_table_names():
            print(f"\n{table}: (not found)")
            continue
        cols = inspector.get_columns(table)
        print(f"\n{table} columns:")
        for c in cols:
            print(f"  {c['name']}: {c['type']}")

    # 3. Sample codes and formats from XEP_FAULTCODES
    if "XEP_FAULTCODES" in inspector.get_table_names():
        with engine.connect() as conn:
            # Distinct codes, sample by format
            r = conn.execute(text("""
                SELECT DISTINCT CODE FROM XEP_FAULTCODES
                WHERE CODE IS NOT NULL AND CODE != ''
                ORDER BY CODE
                LIMIT 500
            """))
            codes = [row[0] for row in r]
            print(f"\nXEP_FAULTCODES: {len(codes)} distinct codes (sample 500)")

            # Analyze formats
            p_codes = [c for c in codes if isinstance(c, str) and c.upper().startswith("P")]
            hex_style = [c for c in codes if isinstance(c, str) and 
                         len(c) >= 3 and c[0].isdigit() and any(h in c.upper() for h in "ABCDEF")]
            numeric = [c for c in codes if isinstance(c, str) and c.isdigit()]
            other = [c for c in codes if c not in p_codes + hex_style + numeric]

            print(f"  P-codes (Pxxxx): {len(p_codes)}")
            if p_codes[:5]:
                print(f"    Examples: {p_codes[:5]}")
            print(f"  Hex-style (2A87, 29CC, etc.): {len(hex_style)}")
            if hex_style[:10]:
                print(f"    Examples: {hex_style[:10]}")
            print(f"  Numeric (420, 171, etc.): {len(numeric)}")
            if numeric[:5]:
                print(f"    Examples: {numeric[:5]}")
            if other:
                print(f"  Other format: {len(other)} - {other[:5]}")

            # Full count
            r2 = conn.execute(text("SELECT COUNT(DISTINCT CODE) FROM XEP_FAULTCODES WHERE CODE IS NOT NULL"))
            total = r2.scalar() or 0
            print(f"\n  Total distinct codes in DB: {total}")

    # 4. Check XEP_COMBINEDFAULTS for P-code / OBD mapping
    if "XEP_COMBINEDFAULTS" in inspector.get_table_names():
        with engine.connect() as conn:
            r = conn.execute(text("SELECT * FROM XEP_COMBINEDFAULTS LIMIT 5"))
            rows = r.fetchall()
            if rows:
                cols = list(r.keys()) if hasattr(r, 'keys') else [f"col{i}" for i in range(len(rows[0]))]
                print(f"\nXEP_COMBINEDFAULTS sample (first 5 rows):")
                for row in rows:
                    print(f"  {dict(zip(cols, row))}")
            else:
                print("\nXEP_COMBINEDFAULTS: (empty)")

    # 5. Export full code list if requested
    if "--export" in sys.argv:
        with engine.connect() as conn:
            r = conn.execute(text("""
                SELECT DISTINCT CODE FROM XEP_FAULTCODES
                WHERE CODE IS NOT NULL AND CODE != ''
                ORDER BY CODE
            """))
            inventory = [{"code": str(row[0]), "title": ""} for row in r]

        out_path = paths.data / "fault_code_inventory.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)
        print(f"\nExported {len(inventory)} codes to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
