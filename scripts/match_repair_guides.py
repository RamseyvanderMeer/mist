#!/usr/bin/env python3
"""
Match repair summaries to repair guides using vector database.

This script processes scraped data that contains repair_summary fields and matches
them to repair guides in the vector database using semantic search. The matched
repair guide information is added to each record.

Uses DB (scraped_records) by default when DATABASE_URL is set. Updates matched_guide_id,
matched_guide_title, match_reasoning in place. Legacy JSONL mode via --input/--output
is deprecated.

Usage:
    python scripts/match_repair_guides.py                    # DB mode when DATABASE_URL set
    python scripts/match_repair_guides.py input.jsonl output.jsonl  # Legacy JSONL
"""
import re
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict
import numpy as np
import yaml
from dotenv import load_dotenv

# Add project root and src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from embeddings.fault_code_encoder import FaultCodeEncoder
from retrieval.vector_store import VectorStore
from paths import get_paths

# Load environment variables
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REASONING_PROMPT = """You are an automotive diagnostic expert. A repair summary from a forum or documentation was matched to a repair guide. Explain why this guide fits and generate verification questions for the diagnosis flow.

Scraped repair summary:
{repair_summary}

Fault codes (if any): {fault_codes}
Symptoms (if any): {symptoms}

Matched repair guide title: {guide_title}
Matched guide text preview: {guide_text}

Output valid JSON only, no markdown:
{{
  "reasoning": "1-3 sentence explanation of why this guide matches the reported fix.",
  "relevance_score": 0.0-1.0,
  "key_symptoms": ["symptom1", "symptom2"],
  "verification_questions": ["Question 1?", "Question 2?", "Question 3?"]
}}

Return only the JSON object."""


class ReasoningGenerator:
    """Generate match reasoning and verification questions using LLM."""

    def __init__(self, llm_config_path: Optional[Path] = None):
        self._client = None
        self._config_path = llm_config_path or (ROOT / "config" / "llm_config.yaml")

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            gemini_config = config.get("gemini", {})
            gemini_config.setdefault("model", "gemini-2.0-flash")
            gemini_config.setdefault("temperature", 0.3)
            gemini_config.setdefault("max_tokens", 512)
            from llm.gemini_client import GeminiClient
            self._client = GeminiClient(gemini_config)
        except Exception as e:
            logger.warning("Could not initialize LLM for reasoning: %s", e)
            self._client = None

    def generate(
        self,
        repair_summary: str,
        guide_title: str,
        guide_text: str,
        fault_codes: Optional[List[str]] = None,
        symptoms: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate match reasoning with reasoning, relevance_score, key_symptoms, verification_questions.

        Returns:
            Dict with reasoning, relevance_score, key_symptoms, verification_questions, or None on failure.
        """
        self._ensure_client()
        if not self._client:
            return None
        fault_codes_str = ", ".join(fault_codes) if fault_codes else "none"
        symptoms_str = symptoms or "none"
        prompt = REASONING_PROMPT.format(
            repair_summary=repair_summary[:2000],
            fault_codes=fault_codes_str,
            symptoms=symptoms_str,
            guide_title=guide_title[:200],
            guide_text=(guide_text or "")[:800],
        )
        try:
            response = self._client.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            text = response.strip()
            json_match = re.search(r"\{[\s\S]*\}", text)
            if not json_match:
                return None
            data = json.loads(json_match.group())
            required = {"reasoning", "relevance_score", "key_symptoms", "verification_questions"}
            if not required.issubset(data.keys()):
                return None
            return {
                "reasoning": str(data["reasoning"]),
                "relevance_score": float(data["relevance_score"]),
                "key_symptoms": list(data["key_symptoms"]) if isinstance(data["key_symptoms"], list) else [],
                "verification_questions": list(data["verification_questions"]) if isinstance(data["verification_questions"], list) else [],
            }
        except Exception as e:
            logger.warning("Reasoning generation failed: %s", e)
            return None


class RepairGuideMatcher:
    """Match repair summaries to repair guides using vector database."""

    def __init__(
        self,
        min_similarity: float = 0.6,
        top_k: int = 5,
        generate_reasoning: bool = False,
    ):
        """
        Initialize matcher.

        Args:
            min_similarity: Minimum similarity score to accept a match (0.0-1.0)
            top_k: Number of candidates to retrieve from vector database
            generate_reasoning: If True, generate match_reasoning (context + verification questions) via LLM
        """
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.generate_reasoning = generate_reasoning
        self._reasoning_generator = ReasoningGenerator() if generate_reasoning else None

        # Initialize encoder and vector store
        logger.info("Initializing FaultCodeEncoder...")
        self.encoder = FaultCodeEncoder()
        
        logger.info("Loading vector store configuration...")
        paths = get_paths()
        with open(paths.retrieval_config, 'r', encoding='utf-8') as f:
            retrieval_config = yaml.safe_load(f)
        
        logger.info("Initializing VectorStore...")
        self.vector_store = VectorStore(retrieval_config.get("vector_store", {}))
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'matched': 0,
            'no_match': 0,
            'high_confidence': 0,  # >= 0.75
            'medium_confidence': 0,  # 0.60-0.75
            'low_confidence': 0,  # < 0.60
            'missing_summary': 0,
            'similarity_scores': [],
            'zero_results': 0,  # Vector search returned no results
            'below_threshold_scores': [],  # Top score when results exist but < threshold (sample)
        }
    
    def match_repair_guide(
        self,
        repair_summary: str,
        fault_codes: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Match repair summary to a repair guide using vector database.
        
        Args:
            repair_summary: Repair summary text from scraped data
            fault_codes: Optional list of fault codes for filtering
        
        Returns:
            Dict with matched repair guide info or None if no good match
        """
        if not repair_summary or not isinstance(repair_summary, str):
            return None
        
        repair_summary = repair_summary.strip()
        if not repair_summary:
            return None
        
        try:
            # Generate embedding for repair summary
            embedding = self.encoder.encode(repair_summary, normalize=True)
            
            # Convert to numpy array (remove batch dimension if single item)
            if embedding.dim() > 1:
                embedding = embedding.squeeze(0)
            embedding_np = embedding.detach().cpu().numpy()
            
            # Search vector database. Use fault_codes filter if available, but fall back to
            # unfiltered search when filter returns 0 results (many repair guides have empty
            # fault_codes or scraped codes may not match ISTA format).
            filter_dict = {"fault_codes": fault_codes} if fault_codes else None
            results = self.vector_store.search(
                query_embedding=embedding_np,
                top_k=self.top_k,
                filter_dict=filter_dict
            )
            # Fallback: if filter returned nothing, retry without filter (semantic match only)
            if not results and filter_dict:
                results = self.vector_store.search(
                    query_embedding=embedding_np,
                    top_k=self.top_k,
                    filter_dict=None
                )
            
            if not results:
                self.stats['zero_results'] += 1
                return None
            
            best_score = results[0]['score']
            if best_score < self.min_similarity:
                # Sample scores for diagnostics (first 200)
                if len(self.stats['below_threshold_scores']) < 200:
                    self.stats['below_threshold_scores'].append(best_score)
                return None
            
            best_match = results[0]
            score = best_match['score']
            self.stats['similarity_scores'].append(score)
            if score >= 0.75:
                self.stats['high_confidence'] += 1
            elif score >= 0.60:
                self.stats['medium_confidence'] += 1
            else:
                self.stats['low_confidence'] += 1
            return {
                'title': best_match.get('title') or best_match.get('procedure_name', ''),
                'procedure_id': best_match.get('procedure_id', ''),
                'similarity_score': score,
                'text': best_match.get('text', '')[:500],  # First 500 chars
                'all_candidates': [
                    {
                        'title': r.get('title') or r.get('procedure_name', ''),
                        'procedure_id': r.get('procedure_id', ''),
                        'score': r['score']
                    }
                    for r in results[:3]  # Top 3 candidates
                ]
            }
            
        except Exception as e:
            logger.warning(f"Error matching repair guide: {e}")
            return None
    
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single record and add matched repair guide.
        
        Args:
            record: Scraped data record with repair_summary field
        
        Returns:
            Record with added repair_guide field
        """
        self.stats['total_processed'] += 1
        
        # Extract repair summary
        repair_summary = record.get('repair_summary', '')
        
        if not repair_summary:
            self.stats['missing_summary'] += 1
            # Keep original record but mark as no match
            record['repair_guide'] = None
            record['match_status'] = 'no_summary'
            return record
        
        # Get fault codes for optional filtering
        fault_codes = record.get('fault_codes', [])
        
        # Match to repair guide
        match = self.match_repair_guide(repair_summary, fault_codes)
        
        if match:
            self.stats['matched'] += 1
            record['repair_guide'] = {
                'title': match['title'],
                'procedure_id': match['procedure_id'],
                'similarity_score': match['similarity_score'],
                'text_preview': match['text'],
                'candidates': match['all_candidates']
            }
            record['match_status'] = 'matched'

            if self.generate_reasoning and self._reasoning_generator:
                reasoning = self._reasoning_generator.generate(
                    repair_summary=repair_summary,
                    guide_title=match['title'],
                    guide_text=match.get('text', ''),
                    fault_codes=fault_codes or None,
                    symptoms=record.get('symptoms'),
                )
                if reasoning:
                    record['match_reasoning'] = reasoning
                else:
                    record['match_reasoning'] = None
        else:
            self.stats['no_match'] += 1
            record['repair_guide'] = None
            record['match_status'] = 'no_match'
        
        return record
    
    def process_from_db(self, db_url: str, unmatched_only: bool = True) -> None:
        """
        Load from scraped_records, match each record, UPDATE rows in place with
        matched_guide_id, matched_guide_title, match_reasoning.
        """
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        logger.info("Loading records from scraped_records...")
        with engine.connect() as conn:
            where = "WHERE repair_summary IS NOT NULL"
            if unmatched_only:
                where += " AND matched_guide_id IS NULL"
            result = conn.execute(
                text(f"""
                    SELECT source_url, repair_summary, fault_codes, symptoms
                    FROM scraped_records {where}
                """)
            )
            columns = result.keys()
            rows = [dict(zip(columns, row)) for row in result]
        
        logger.info(f"Processing {len(rows)} records...")
        for i, row in enumerate(rows):
            fault_codes = row.get("fault_codes")
            if isinstance(fault_codes, str):
                try:
                    fault_codes = json.loads(fault_codes) if fault_codes else []
                except json.JSONDecodeError:
                    fault_codes = []
            record = {
                "source_url": row["source_url"],
                "repair_summary": row["repair_summary"],
                "fault_codes": fault_codes if isinstance(fault_codes, list) else [],
                "symptoms": row.get("symptoms"),
            }
            processed = self.process_record(record)
            match_guide = processed.get("repair_guide")
            match_reasoning = processed.get("match_reasoning")
            matched_id = match_guide.get("procedure_id") if match_guide else None
            matched_title = match_guide.get("title") if match_guide else None
            reasoning_json = json.dumps(match_reasoning, ensure_ascii=False) if match_reasoning else None
            with engine.connect() as conn:
                conn.execute(
                    text("""
                        UPDATE scraped_records
                        SET matched_guide_id = :mid, matched_guide_title = :title,
                            match_reasoning = CAST(:reasoning AS jsonb)
                        WHERE source_url = :url
                    """),
                    {"mid": matched_id, "title": matched_title, "reasoning": reasoning_json, "url": row["source_url"]}
                )
                conn.commit()
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(rows)} records...")
        logger.info(f"Completed. Matched {self.stats['matched']}/{self.stats['total_processed']} records.")
    
    def process_file(self, input_path: Path, output_path: Path) -> None:
        """
        Process a JSONL file.
        
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
                    processed_records.append(processed)
                    
                    # Log progress every 100 records
                    if line_num % 100 == 0:
                        logger.info(f"Processed {line_num} records...")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
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
        """Print matching statistics."""
        logger.info("=" * 60)
        logger.info("Matching Statistics")
        logger.info("=" * 60)
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Matched: {self.stats['matched']} ({self.stats['matched']/max(self.stats['total_processed'], 1)*100:.1f}%)")
        logger.info(f"No match: {self.stats['no_match']} ({self.stats['no_match']/max(self.stats['total_processed'], 1)*100:.1f}%)")
        logger.info(f"Missing summary: {self.stats['missing_summary']}")
        zero = self.stats.get('zero_results', 0)
        below = self.stats.get('below_threshold_scores', [])
        if zero or below:
            logger.info("")
            logger.info("Diagnostics (why no match):")
            if zero:
                logger.info(f"  Queries with 0 vector results: {zero}")
            if below:
                logger.info(f"  Queries with results but score < {self.min_similarity}: {len(below)} (sample)")
                logger.info(f"    Score range: min={min(below):.3f}, max={max(below):.3f}, mean={np.mean(below):.3f}")
                logger.info(f"    Try --min-similarity {max(below):.2f} to accept some matches")
        logger.info("")
        logger.info("Confidence breakdown:")
        logger.info(f"  High confidence (>=0.75): {self.stats['high_confidence']}")
        logger.info(f"  Medium confidence (0.60-0.75): {self.stats['medium_confidence']}")
        logger.info(f"  Low confidence (<0.60): {self.stats['low_confidence']}")
        
        if self.stats['similarity_scores']:
            scores = self.stats['similarity_scores']
            logger.info("")
            logger.info("Similarity score statistics:")
            logger.info(f"  Mean: {np.mean(scores):.3f}")
            logger.info(f"  Median: {np.median(scores):.3f}")
            logger.info(f"  Min: {np.min(scores):.3f}")
            logger.info(f"  Max: {np.max(scores):.3f}")


def main():
    """Main entry point. Uses DB by default when DATABASE_URL is set."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Match repair summaries to repair guides. Uses DB by default when DATABASE_URL is set."
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
        '--min-similarity',
        type=float,
        default=0.6,
        help='Minimum similarity score to accept a match (0.0-1.0, default: 0.6)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of candidates to retrieve from vector database (default: 5)'
    )
    parser.add_argument(
        '--generate-reasoning',
        action='store_true',
        help='Generate match_reasoning (context + verification questions) via LLM for diagnosis flow'
    )
    parser.add_argument(
        '--from-db',
        action='store_true',
        help='Explicitly use DB (default when DATABASE_URL is set)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all records (default: only unmatched, i.e. matched_guide_id IS NULL)'
    )

    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    use_db = bool(db_url and db_url.startswith("postgresql")) and (
        args.from_db or (args.input is None and args.output is None)
    )

    matcher = RepairGuideMatcher(
        min_similarity=args.min_similarity,
        top_k=args.top_k,
        generate_reasoning=args.generate_reasoning,
    )

    if use_db:
        if not db_url or not db_url.startswith("postgresql"):
            logger.error("DATABASE_URL required for DB mode. Set in .env or export.")
            sys.exit(1)
        matcher.process_from_db(db_url, unmatched_only=not args.all)
        matcher.print_statistics()
    elif args.input is not None and args.output is not None:
        logger.warning("JSONL input/output is deprecated. Use DB mode (set DATABASE_URL) for DB-first workflow.")
        if args.input.is_file():
            matcher.process_file(args.input, args.output)
            matcher.print_statistics()
        elif args.input.is_dir():
            matcher.process_directory(args.input, args.output)
        else:
            logger.error(f"Input path does not exist: {args.input}")
            sys.exit(1)
    else:
        logger.error(
            "Either set DATABASE_URL for DB mode, or provide both input and output for legacy JSONL mode."
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
