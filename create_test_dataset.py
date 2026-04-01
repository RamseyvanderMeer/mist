#!/usr/bin/env python3
"""
Create a comprehensive test dataset for MIST fault code matching evaluation.

This script extracts high-quality records from the Neon database and creates
a curated test set with known good matches for evaluation.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection (never hardcode credentials — use DATABASE_URL from the environment)
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    print("DATABASE_URL must be set", file=sys.stderr)
    sys.exit(1)

# Test set configuration
TEST_SET_SIZE = 50  # Small, high-quality test set
OUTPUT_FILE = Path(__file__).parent / "test_dataset.json"


def get_db_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(DB_URL)


def fetch_high_quality_records(min_quality_score: float = 0.7, limit: int = 200) -> List[Dict]:
    """
    Fetch records with high quality scores and complete data.
    
    Criteria:
    - Has fault codes
    - Has repair summary or repair guide
    - Has matched_guide_id (ground truth)
    - Quality score >= min_quality_score
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
        SELECT 
            id,
            fault_codes,
            symptoms,
            repair_summary,
            repair_guide,
            matched_guide_id,
            matched_guide_title,
            obd_data,
            vehicle_context,
            quality_score,
            confidence_score,
            source_url,
            source_type,
            outcome
        FROM scraped_records
        WHERE 
            fault_codes IS NOT NULL 
            AND fault_codes != '[]'
            AND fault_codes != ''
            AND (repair_summary IS NOT NULL OR repair_guide IS NOT NULL)
            AND matched_guide_id IS NOT NULL
            AND quality_score >= %s
        ORDER BY quality_score DESC, confidence_score DESC
        LIMIT %s;
    """
    
    cursor.execute(query, (min_quality_score, limit))
    records = cursor.fetchall()
    conn.close()
    
    return [dict(r) for r in records]


def fetch_records_by_fault_code_popularity(limit: int = 100) -> List[Dict]:
    """
    Fetch records grouped by common fault codes to ensure coverage.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # First, find common fault codes
    query = """
        SELECT fault_codes, COUNT(*) as count
        FROM scraped_records
        WHERE fault_codes IS NOT NULL 
            AND fault_codes != '[]'
            AND fault_codes != ''
            AND matched_guide_id IS NOT NULL
        GROUP BY fault_codes
        ORDER BY count DESC
        LIMIT 20;
    """
    
    cursor.execute(query)
    common_codes = cursor.fetchall()
    
    records = []
    for code_row in common_codes:
        fault_codes = code_row['fault_codes']
        
        query = """
            SELECT 
                id,
                fault_codes,
                symptoms,
                repair_summary,
                repair_guide,
                matched_guide_id,
                matched_guide_title,
                obd_data,
                vehicle_context,
                quality_score,
                confidence_score,
                source_url,
                source_type,
                outcome
            FROM scraped_records
            WHERE fault_codes = %s
                AND matched_guide_id IS NOT NULL
                AND quality_score >= 0.6
            ORDER BY quality_score DESC
            LIMIT 3;
        """
        cursor.execute(query, (fault_codes,))
        code_records = cursor.fetchall()
        records.extend([dict(r) for r in code_records])
    
    conn.close()
    return records


def create_diverse_test_set(size: int = TEST_SET_SIZE) -> List[Dict]:
    """
    Create a diverse test set with:
    - High quality records with ground truth
    - Various fault code combinations
    - Different vehicle contexts
    - Mix of single and multiple fault codes
    """
    print("Fetching high-quality records...")
    high_quality = fetch_high_quality_records(min_quality_score=0.7, limit=100)
    print(f"  Found {len(high_quality)} high-quality records")
    
    print("Fetching records by fault code popularity...")
    by_popularity = fetch_records_by_fault_code_popularity(limit=60)
    print(f"  Found {len(by_popularity)} records by popularity")
    
    # Combine and deduplicate
    all_records = {r['id']: r for r in high_quality + by_popularity}
    unique_records = list(all_records.values())
    
    print(f"Total unique records: {len(unique_records)}")
    
    # Categorize records
    single_code = [r for r in unique_records if len(json.loads(r['fault_codes'] or '[]')) == 1]
    multi_code = [r for r in unique_records if len(json.loads(r['fault_codes'] or '[]')) > 1]
    with_symptoms = [r for r in unique_records if r.get('symptoms')]
    with_obd = [r for r in unique_records if r.get('obd_data')]
    
    print(f"  Single fault code: {len(single_code)}")
    print(f"  Multiple fault codes: {len(multi_code)}")
    print(f"  With symptoms: {len(with_symptoms)}")
    print(f"  With OBD data: {len(with_obd)}")
    
    # Create diverse sample
    test_set = []
    
    # Add diverse selection
    categories = [
        (single_code[:10], "single_code"),
        (multi_code[:10], "multi_code"),
        (with_symptoms[:10], "with_symptoms"),
        (with_obd[:5], "with_obd_data"),
    ]
    
    for cat_records, cat_name in categories:
        for record in cat_records:
            if len(test_set) < size:
                record['_test_category'] = cat_name
                test_set.append(record)
    
    # Fill remaining with highest quality
    remaining = size - len(test_set)
    if remaining > 0:
        for record in high_quality:
            if record['id'] not in [r['id'] for r in test_set] and len(test_set) < size:
                record['_test_category'] = 'high_quality'
                test_set.append(record)
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(test_set)
    
    return test_set


def enrich_test_records(records: List[Dict]) -> List[Dict]:
    """
    Enrich test records with additional metadata for evaluation.
    """
    enriched = []
    
    for record in records:
        # Parse fault codes
        try:
            fault_codes = json.loads(record['fault_codes'] or '[]')
        except:
            fault_codes = []
        
        # Build query texts
        symptom_text = record.get('symptoms', '') or ''
        repair_text = record.get('repair_summary', '') or record.get('repair_guide', '') or ''
        
        query_text_symptom = f"Fault codes: {', '.join(fault_codes)}"
        if symptom_text:
            query_text_symptom += f". Problem: {symptom_text[:200]}"
        
        # Ground truth
        ground_truth = {
            'guide_id': record.get('matched_guide_id'),
            'guide_title': record.get('matched_guide_title'),
            'repair_text': repair_text[:500] if repair_text else None,
        }
        
        # Evaluation metadata
        eval_metadata = {
            'num_fault_codes': len(fault_codes),
            'has_symptoms': bool(symptom_text),
            'has_obd_data': bool(record.get('obd_data')),
            'has_vehicle_context': bool(record.get('vehicle_context')),
            'quality_score': record.get('quality_score'),
            'confidence_score': record.get('confidence_score'),
            'test_category': record.get('_test_category', 'general'),
        }
        
        enriched_record = {
            'id': record['id'],
            'fault_codes': fault_codes,
            'query_text_symptom': query_text_symptom,
            'query_text_solution': f"Fix: {repair_text[:300]}" if repair_text else query_text_symptom,
            'ground_truth': ground_truth,
            'evaluation': eval_metadata,
            'source': {
                'url': record.get('source_url'),
                'type': record.get('source_type'),
            },
            'raw_data': {
                'symptoms': symptom_text,
                'obd_data': record.get('obd_data'),
                'vehicle_context': record.get('vehicle_context'),
                'repair_summary': record.get('repair_summary'),
                'repair_guide': record.get('repair_guide'),
            }
        }
        
        enriched.append(enriched_record)
    
    return enriched


def save_test_dataset(records: List[Dict], output_file: Path):
    """Save test dataset to JSON file."""
    dataset = {
        'metadata': {
            'created': str(pd.Timestamp.now()) if 'pd' in dir() else '2024-01-01',
            'total_records': len(records),
            'description': 'MIST fault code matching test dataset',
            'source': 'Neon PostgreSQL scraped_records table',
        },
        'records': records,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nTest dataset saved to: {output_file}")
    print(f"Total records: {len(records)}")


def print_dataset_stats(records: List[Dict]):
    """Print statistics about the test dataset."""
    print("\n" + "="*60)
    print("TEST DATASET STATISTICS")
    print("="*60)
    
    categories = {}
    fault_code_counts = {}
    has_symptoms = 0
    has_obd = 0
    has_vehicle = 0
    
    for r in records:
        cat = r['evaluation']['test_category']
        categories[cat] = categories.get(cat, 0) + 1
        
        num_codes = r['evaluation']['num_fault_codes']
        fault_code_counts[num_codes] = fault_code_counts.get(num_codes, 0) + 1
        
        if r['evaluation']['has_symptoms']:
            has_symptoms += 1
        if r['evaluation']['has_obd_data']:
            has_obd += 1
        if r['evaluation']['has_vehicle_context']:
            has_vehicle += 1
    
    print(f"\nBy Category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print(f"\nBy Number of Fault Codes:")
    for num, count in sorted(fault_code_counts.items()):
        print(f"  {num} code(s): {count}")
    
    print(f"\nData Completeness:")
    print(f"  Has symptoms: {has_symptoms}/{len(records)} ({100*has_symptoms/len(records):.1f}%)")
    print(f"  Has OBD data: {has_obd}/{len(records)} ({100*has_obd/len(records):.1f}%)")
    print(f"  Has vehicle context: {has_vehicle}/{len(records)} ({100*has_vehicle/len(records):.1f}%)")
    
    print(f"\nSample Fault Code Combinations:")
    for r in records[:5]:
        codes = ', '.join(r['fault_codes'])
        print(f"  {codes} -> {r['ground_truth']['guide_title'][:60]}...")


def main():
    print("="*60)
    print("Creating MIST Test Dataset")
    print("="*60)
    
    # Create diverse test set
    test_records = create_diverse_test_set(size=TEST_SET_SIZE)
    
    # Enrich with evaluation metadata
    enriched_records = enrich_test_records(test_records)
    
    # Print statistics
    print_dataset_stats(enriched_records)
    
    # Save to file
    save_test_dataset(enriched_records, OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("Test dataset creation complete!")
    print("="*60)
    print(f"\nTo use this dataset:")
    print(f"  python3 -c \"import json; data=json.load(open('{OUTPUT_FILE}')); print(len(data['records']))\"")


if __name__ == "__main__":
    main()
