#!/usr/bin/env python3
"""
Process and validate scraped web data for MIST training.

This script:
1. Validates scraped data format and quality
2. Normalizes OBD-II data
3. Extracts and structures repair procedures
4. Calculates quality scores
5. Deduplicates records
6. Outputs cleaned data in MIST-compatible format

Uses DB (scraped_records) by default when DATABASE_URL is set.
Legacy JSONL mode via --input/--output is deprecated.
"""
import sys
from pathlib import Path

# Add project root to path before importing scrapers
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging
from datetime import datetime, timezone

from scrapers.utils.constants import FAULT_CODE_VALIDATE_PATTERNS, OBD_RANGES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScrapedDataProcessor:
    """Process and validate scraped web data for MIST training."""

    def __init__(self, min_quality_score: float = 0.6):
        """
        Initialize processor.
        
        Args:
            min_quality_score: Minimum quality score to accept (0.0-1.0)
        """
        self.min_quality_score = min_quality_score
        self.stats = {
            'total_processed': 0,
            'valid': 0,
            'invalid': 0,
            'duplicates': 0,
            'quality_rejected': 0,
            'fault_codes': defaultdict(int),
            'cause_to_solution': 0,
            'vehicles': defaultdict(int),
        }
    
    def validate_fault_code(self, code: str) -> bool:
        """
        Validate fault code format.
        
        Args:
            code: Fault code string
            
        Returns:
            True if valid format
        """
        if not code or not isinstance(code, str):
            return False
        
        code = code.strip().upper()

        for pattern in FAULT_CODE_VALIDATE_PATTERNS:
            if pattern.match(code):
                return True

        return False
    
    def normalize_obd_data(self, obd_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize and validate OBD-II data.
        
        Args:
            obd_data: Raw OBD data dictionary
            
        Returns:
            Normalized OBD data dictionary
        """
        if not obd_data or not isinstance(obd_data, dict):
            return {}
        
        normalized = {}
        
        for key, value in obd_data.items():
            # Normalize key names
            key_lower = key.lower().replace(' ', '_').replace('-', '_')
            
            # Convert to float if possible
            try:
                float_value = float(value)
                
                # Validate range if known
                if key_lower in OBD_RANGES:
                    min_val, max_val = OBD_RANGES[key_lower]
                    if min_val <= float_value <= max_val:
                        normalized[key_lower] = float_value
                    else:
                        logger.debug(f"OBD value out of range: {key_lower}={float_value}")
                else:
                    # Unknown parameter, accept if reasonable
                    normalized[key_lower] = float_value
                    
            except (ValueError, TypeError):
                logger.debug(f"Could not convert OBD value to float: {key}={value}")
                continue
        
        return normalized
    
    def extract_repair_steps(self, text: str) -> List[str]:
        """
        Extract structured repair steps from free text.
        
        Args:
            text: Free text containing repair procedure
            
        Returns:
            List of repair step strings
        """
        if not text:
            return []
        
        # Try to extract numbered steps
        numbered_pattern = r'(?:^|\n)\s*(\d+)[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|$)'
        matches = re.findall(numbered_pattern, text, re.MULTILINE)
        
        if matches:
            return [step.strip() for _, step in matches]
        
        # Try to extract bullet points
        bullet_pattern = r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|$)'
        matches = re.findall(bullet_pattern, text, re.MULTILINE)
        
        if matches:
            return [step.strip() for step in matches]
        
        # Fallback: split by newlines and filter
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Filter out very short lines (likely not steps)
        steps = [line for line in lines if len(line) > 20]
        
        return steps[:10]  # Limit to 10 steps
    
    def calculate_quality_score(self, record: Dict[str, Any]) -> float:
        """
        Calculate data quality score (0.0-1.0).
        
        Scoring:
        - Fault codes: 0.3
        - Vehicle context: 0.2
        - Repair guide: 0.3
        - OBD data: 0.15
        - Outcome: 0.05
        
        Args:
            record: Data record dictionary
            
        Returns:
            Quality score (0.0-1.0)
        """
        score = 0.0
        
        # Fault codes (0.3) or symptoms for cause_to_solution (0.2)
        record_type = record.get('record_type', 'fault_code')
        fault_codes = record.get('fault_codes', [])
        if fault_codes and any(self.validate_fault_code(code) for code in fault_codes):
            score += 0.3
        elif record_type == 'cause_to_solution':
            symptoms = record.get('symptoms', '')
            if symptoms and len(symptoms) >= 20:
                score += 0.2
            elif symptoms:
                score += 0.1
        
        # Vehicle context (0.2)
        vehicle_context = record.get('vehicle_context', {})
        if vehicle_context:
            has_make = bool(vehicle_context.get('make'))
            has_model = bool(vehicle_context.get('model'))
            has_year = bool(vehicle_context.get('year'))
            
            if has_make and has_model and has_year:
                score += 0.2
            elif (has_make and has_model) or (has_make and has_year):
                score += 0.1
        
        # Repair guide / repair_summary (0.3)
        repair_guide = record.get('repair_guide', {})
        repair_summary = record.get('repair_summary', '')
        if repair_guide or repair_summary:
            if isinstance(repair_guide, dict):
                has_title = bool(repair_guide.get('title'))
                has_steps = bool(repair_guide.get('procedure_steps') or repair_guide.get('steps'))
            else:
                has_title = bool(repair_guide)
                has_steps = False
            has_summary = bool(repair_summary) and len(str(repair_summary)) >= 50
            if has_title and (has_steps or has_summary):
                score += 0.3
            elif has_title or has_steps or has_summary:
                score += 0.15
        
        # OBD data (0.15)
        obd_data = record.get('obd_data', {})
        if obd_data:
            num_params = len(obd_data)
            if num_params >= 5:
                score += 0.15
            elif num_params >= 3:
                score += 0.1
            elif num_params >= 1:
                score += 0.05
        
        # Outcome (0.05)
        outcome = record.get('outcome')
        if outcome and outcome.lower() in ['success', 'failure', 'partial']:
            score += 0.05
        
        return score
    
    def deduplicate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate records.
        
        Two records are considered duplicates if they have:
        - Same fault codes
        - Same vehicle context (make/model/year)
        - Similar repair guide title
        
        Args:
            records: List of records
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique_records = []
        
        for record in records:
            fault_codes = tuple(sorted(record.get('fault_codes', [])))
            vehicle = record.get('vehicle_context', {})
            vehicle_key = (
                vehicle.get('make', ''),
                vehicle.get('model', ''),
                vehicle.get('year', '')
            )
            repair_guide = record.get('repair_guide', {})
            repair_title = repair_guide.get('title', '') if isinstance(repair_guide, dict) else str(repair_guide)
            symptoms = record.get('symptoms', '')[:50] if record.get('record_type') == 'cause_to_solution' else ''
            signature = (fault_codes, vehicle_key, repair_title.lower()[:50], symptoms)
            
            if signature not in seen:
                seen.add(signature)
                unique_records.append(record)
            else:
                self.stats['duplicates'] += 1
        
        return unique_records
    
    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single record.
        
        Args:
            record: Raw scraped data record
            
        Returns:
            Processed record or None if invalid
        """
        self.stats['total_processed'] += 1

        record_type = record.get('record_type', 'fault_code')
        fault_codes = record.get('fault_codes', [])
        valid_fault_codes = [
            code.strip().upper()
            for code in fault_codes
            if self.validate_fault_code(code)
        ]

        # Cause-to-solution: no fault codes required, but need symptoms + repair_summary
        if record_type == 'cause_to_solution':
            symptoms = (record.get('symptoms') or '').strip()
            repair_summary = (record.get('repair_summary') or record.get('repair_guide') or '')
            if isinstance(repair_summary, dict):
                repair_summary = repair_summary.get('title', '') or repair_summary.get('description', '')
            repair_summary = str(repair_summary).strip()
            if not repair_summary or len(repair_summary) < 50:
                logger.debug("Cause-to-solution record missing substantial repair_summary")
                self.stats['invalid'] += 1
                return None
            valid_fault_codes = []  # Empty for cause_to_solution
        elif not fault_codes:
            logger.debug("Record missing fault codes")
            self.stats['invalid'] += 1
            return None
        elif not valid_fault_codes:
            logger.debug(f"Record has no valid fault codes: {fault_codes}")
            self.stats['invalid'] += 1
            return None
        
        # Normalize OBD data
        obd_data = self.normalize_obd_data(record.get('obd_data', {}))
        
        # Process repair guide (fallback to repair_summary for legacy/external files)
        repair_guide = record.get('repair_guide') or record.get('repair_summary') or {}
        if isinstance(repair_guide, str):
            # Convert string to dict
            repair_guide = {
                'title': repair_guide[:200],
                'procedure_steps': self.extract_repair_steps(repair_guide)
            }
        elif isinstance(repair_guide, dict):
            # Extract steps if not already structured
            if 'procedure_steps' not in repair_guide and 'steps' not in repair_guide:
                text = repair_guide.get('title', '') + ' ' + repair_guide.get('description', '')
                repair_guide['procedure_steps'] = self.extract_repair_steps(text)
        
        # Build processed record
        processed = {
            'fault_codes': valid_fault_codes,
            'record_type': record_type,
            'obd_data': obd_data,
            'vehicle_context': record.get('vehicle_context', {}),
            'repair_guide': repair_guide,
            'outcome': record.get('outcome', 'unknown'),
            'source_url': record.get('source_url', ''),
            'timestamp': record.get('timestamp', datetime.now(timezone.utc).isoformat()),
            'quality_score': 0.0  # Will be calculated
        }
        if record_type == 'cause_to_solution':
            processed['symptoms'] = (record.get('symptoms') or '').strip()
            rs = record.get('repair_summary') or ''
            processed['repair_summary'] = rs if isinstance(rs, str) else str(repair_guide.get('title', '') if isinstance(repair_guide, dict) else repair_guide)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(processed)
        processed['quality_score'] = quality_score
        
        # Check quality threshold
        if quality_score < self.min_quality_score:
            logger.debug(f"Record rejected: quality_score={quality_score:.2f} < {self.min_quality_score}")
            self.stats['quality_rejected'] += 1
            return None
        
        # Update statistics
        self.stats['valid'] += 1
        if record_type == 'cause_to_solution':
            self.stats['cause_to_solution'] += 1
        for code in valid_fault_codes:
            self.stats['fault_codes'][code] += 1
        
        vehicle = processed.get('vehicle_context', {})
        if vehicle.get('make') and vehicle.get('model'):
            vehicle_key = f"{vehicle['make']} {vehicle['model']}"
            self.stats['vehicles'][vehicle_key] += 1
        
        return processed
    
    def _row_to_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DB row to record dict for process_record()."""
        def _parse_json(val: Any, default: Any) -> Any:
            if val is None:
                return default
            if isinstance(val, (list, dict)):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return default
            return default

        return {
            "source_url": row.get("source_url"),
            "fault_codes": _parse_json(row.get("fault_codes"), []),
            "obd_data": _parse_json(row.get("obd_data"), {}),
            "vehicle_context": _parse_json(row.get("vehicle_context"), {}),
            "repair_summary": row.get("repair_summary"),
            "repair_guide": row.get("repair_guide") or row.get("repair_summary"),
            "outcome": row.get("outcome"),
            "source_type": row.get("source_type", "forum"),
            "record_type": row.get("record_type", "fault_code"),
            "symptoms": row.get("symptoms"),
            "timestamp": str(row.get("timestamp")) if row.get("timestamp") else None,
        }
    
    def process_from_db(self, db_url: str) -> List[Dict[str, Any]]:
        """Load from scraped_records, process, deduplicate, and return valid records."""
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        logger.info("Loading records from scraped_records...")
        records = []
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT source_url, fault_codes, obd_data, vehicle_context,
                           repair_summary, outcome, source_type, record_type,
                           symptoms, timestamp
                    FROM scraped_records
                    WHERE repair_summary IS NOT NULL
                      AND (record_type = 'cause_to_solution'
                           OR (fault_codes IS NOT NULL AND fault_codes::text NOT IN ('[]', 'null')))
                """)
            )
            columns = result.keys()
            for row in result:
                rec = dict(zip(columns, row))
                record = self._row_to_record(rec)
                processed = self.process_record(record)
                if processed:
                    records.append(processed)
        logger.info(f"Processed {len(records)} valid records from DB")
        records = self.deduplicate_records(records)
        logger.info(f"After deduplication: {len(records)} records")
        return records
    
    def write_to_db(self, records: List[Dict[str, Any]], db_url: str) -> None:
        """Update scraped_records with quality_score for processed records."""
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        updated = 0
        with engine.connect() as conn:
            for rec in records:
                source_url = rec.get("source_url")
                if not source_url:
                    continue
                quality_score = rec.get("quality_score")
                if quality_score is None:
                    continue
                conn.execute(
                    text("""
                        UPDATE scraped_records
                        SET quality_score = :quality_score
                        WHERE source_url = :source_url
                    """),
                    {"quality_score": quality_score, "source_url": source_url}
                )
                updated += 1
            conn.commit()
        logger.info(f"Updated quality_score for {updated} records in scraped_records")
    
    def process_file(self, input_path: Path, output_path: Path) -> None:
        """
        Process a JSONL file (legacy; deprecated when using DB).
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
        """
        logger.info(f"Processing file: {input_path}")
        
        processed_records = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    processed = self.process_record(record)
                    if processed:
                        processed_records.append(processed)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
        # Deduplicate
        logger.info(f"Deduplicating {len(processed_records)} records...")
        processed_records = self.deduplicate_records(processed_records)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in processed_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Processed {len(processed_records)} records -> {output_path}")
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """
        Process all JSONL files in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
        """
        logger.info(f"Processing directory: {input_dir}")
        
        jsonl_files = list(input_dir.glob('*.jsonl'))
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        for input_file in jsonl_files:
            output_file = output_dir / input_file.name
            self.process_file(input_file, output_file)
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self) -> None:
        """Print processing statistics."""
        logger.info("=" * 60)
        logger.info("Processing Statistics")
        logger.info("=" * 60)
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Valid: {self.stats['valid']}")
        logger.info(f"Invalid: {self.stats['invalid']}")
        logger.info(f"Duplicates removed: {self.stats['duplicates']}")
        logger.info(f"Quality rejected: {self.stats['quality_rejected']}")
        logger.info(f"Unique fault codes: {len(self.stats['fault_codes'])}")
        logger.info(f"Cause-to-solution records: {self.stats['cause_to_solution']}")
        logger.info(f"Unique vehicles: {len(self.stats['vehicles'])}")
        
        # Top fault codes
        if self.stats['fault_codes']:
            logger.info("\nTop 10 fault codes:")
            for code, count in sorted(
                self.stats['fault_codes'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                logger.info(f"  {code}: {count}")


def main():
    """Main entry point. Uses DB by default when DATABASE_URL is set."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process scraped web data for MIST training. Uses DB by default when DATABASE_URL is set."
    )
    parser.add_argument(
        'input',
        type=Path,
        nargs='?',
        default=None,
        help='Input file or directory (JSONL format). Omit to use DB when DATABASE_URL is set.'
    )
    parser.add_argument(
        'output',
        type=Path,
        nargs='?',
        default=None,
        help='Output file or directory (JSONL format). Omit when using DB mode.'
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.6,
        help='Minimum quality score (0.0-1.0, default: 0.6)'
    )
    parser.add_argument(
        '--from-db',
        action='store_true',
        help='Explicitly use DB (default when DATABASE_URL is set)'
    )
    
    args = parser.parse_args()
    
    db_url = os.environ.get("DATABASE_URL", "")
    use_db = bool(db_url and db_url.startswith("postgresql")) and (args.from_db or (args.input is None and args.output is None))
    
    if use_db:
        if not db_url or not db_url.startswith("postgresql"):
            logger.error("DATABASE_URL required for DB mode. Set in .env or export.")
            sys.exit(1)
        processor = ScrapedDataProcessor(min_quality_score=args.min_quality)
        records = processor.process_from_db(db_url)
        processor.write_to_db(records, db_url)
        processor.print_statistics()
    elif args.input is not None and args.output is not None:
        logger.warning("JSONL input/output is deprecated. Use DB mode (set DATABASE_URL) for DB-first workflow.")
        processor = ScrapedDataProcessor(min_quality_score=args.min_quality)
        if args.input.is_file():
            processor.process_file(args.input, args.output)
        elif args.input.is_dir():
            processor.process_directory(args.input, args.output)
        else:
            logger.error(f"Input path does not exist: {args.input}")
            sys.exit(1)
    else:
        logger.error(
            "Either set DATABASE_URL for DB mode, or provide both --input and --output for legacy JSONL mode."
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
