"""
Test fault code to guide matching accuracy.
Compares performance with and without P-code to hex mappings.
"""
import json
import sqlite3
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_test_dataset():
    """Load test dataset."""
    with open(ROOT / "test_dataset.json") as f:
        data = json.load(f)
    return data.get("records", [])


def get_hex_mappings_for_pcode(pcode: str) -> list:
    """Get hex mappings for a P-code from database."""
    db_path = ROOT / "data" / "databases" / "mist_data.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT hex_codes FROM bmwfault_mappings WHERE pcode = ?",
        (pcode.upper(),)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0].split(",")
    return []


def test_matching_accuracy():
    """Test matching accuracy with and without hex mappings."""
    print("=" * 70)
    print("FAULT CODE TO GUIDE MATCHING ACCURACY TEST")
    print("=" * 70)
    
    # Load test dataset
    test_data = load_test_dataset()
    print(f"\nTest dataset: {len(test_data)} records")
    
    # Check which test records have P-codes with mappings
    records_with_mappings = 0
    records_without_mappings = 0
    total_hex_codes = 0
    
    print("\n" + "-" * 70)
    print("P-CODE MAPPING COVERAGE:")
    print("-" * 70)
    
    for record in test_data[:10]:  # Show first 10
        fault_codes = record.get("fault_codes", [])
        all_hex = []
        
        for pcode in fault_codes:
            hex_codes = get_hex_mappings_for_pcode(pcode)
            if hex_codes:
                all_hex.extend(hex_codes)
        
        has_mapping = len(all_hex) > 0
        if has_mapping:
            records_with_mappings += 1
            total_hex_codes += len(all_hex)
            status = f"✓ {len(all_hex)} hex codes"
        else:
            records_without_mappings += 1
            status = "✗ No mappings"
        
        pcode_str = ", ".join(fault_codes)
        print(f"  {pcode_str:<25} -> {status}")
    
    if len(test_data) > 10:
        print(f"  ... and {len(test_data) - 10} more records")
    
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print("-" * 70)
    print(f"  Records with hex mappings:    {records_with_mappings} / {min(10, len(test_data))} ({records_with_mappings/min(10, len(test_data))*100:.1f}%)")
    print(f"  Records without mappings:     {records_without_mappings} / {min(10, len(test_data))}")
    print(f"  Total hex codes found:        {total_hex_codes}")
    
    # Overall database stats
    print("\n" + "-" * 70)
    print("DATABASE COVERAGE:")
    print("-" * 70)
    
    conn = sqlite3.connect(ROOT / "data" / "databases" / "mist_data.db")
    cursor = conn.execute("SELECT COUNT(*) FROM bmwfault_mappings")
    total_mappings = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT COUNT(DISTINCT pcode) FROM bmwfault_pcodes")
    total_pcodes = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT COUNT(*) FROM bmwfault_pcodes")
    total_rows = cursor.fetchone()[0]
    conn.close()
    
    print(f"  Total P-codes with mappings:  {total_mappings}")
    print(f"  Total unique P-codes:         {total_pcodes}")
    print(f"  Total P-code -> hex rows:     {total_rows:,}")
    print(f"  Average hex codes per P-code: {total_rows/total_pcodes:.1f}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR FULL EVALUATION:")
    print("=" * 70)
    print("1. Run retrieval tests with P-codes only")
    print("2. Run retrieval tests with P-codes + hex codes")
    print("3. Compare precision@k and recall metrics")
    print("4. Analyze improvement from hex mappings")


if __name__ == "__main__":
    test_matching_accuracy()
