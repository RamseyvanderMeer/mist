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
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
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
from llm.openai_client import OpenAIClient

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
                config = yaml.safe_load(f) or {}
            openai_config = dict(config.get("openai", {}) or {})
            openai_config["api_key_env"] = "OPENAI_API_KEY"
            openai_config["model"] = os.getenv(
                "OPENAI_MODEL", openai_config.get("model", "gpt-4o")
            )
            openai_config.setdefault("temperature", 0.3)
            openai_config.setdefault("max_tokens", 512)
            self._client = OpenAIClient(openai_config)
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
        llm_gating: bool = False,
        llm_min_confidence: float = 0.7,
        log_every: int = 25,
        verbose: bool = False,
    ):
        """
        Initialize matcher.

        Args:
            min_similarity: Minimum similarity score to accept a match (0.0-1.0)
            top_k: Number of candidates to retrieve from vector database
            generate_reasoning: If True, generate match_reasoning (context + verification questions) via LLM
            llm_gating: If True, require LLM relevance score to accept a match.
            llm_min_confidence: Minimum relevance score required when llm_gating is enabled.
        """
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.generate_reasoning = generate_reasoning
        self.llm_gating = llm_gating
        self.llm_min_confidence = max(0.0, min(1.0, llm_min_confidence))
        self.log_every = max(1, int(log_every))
        self.verbose = verbose
        self._reasoning_generator = (
            ReasoningGenerator() if (generate_reasoning or llm_gating) else None
        )

        # Initialize encoder and vector store (must match indexer config for dimension)
        paths = get_paths()
        logger.info("Loading embedding configuration...")
        with open(paths.embedding_config, 'r', encoding='utf-8') as f:
            embedding_config = yaml.safe_load(f)
        fc_config = embedding_config.get("models", {}).get("fault_code", {})
        logger.info("Initializing FaultCodeEncoder...")
        self.encoder = FaultCodeEncoder(
            model_name=fc_config.get("model_name", "intfloat/e5-mistral-7b-instruct"),
            device=fc_config.get("device", "auto"),
            projection_dim=fc_config.get("projection_dim", 768),
        )

        logger.info("Loading vector store configuration...")
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
            'llm_gating_enabled': 0,
            'llm_gating_passed': 0,
            'llm_gating_rejected': 0,
            'llm_gating_failures': 0,
        }
    
    def match_repair_guide(
        self,
        repair_summary: str,
        fault_codes: Optional[List[str]] = None,
        symptoms: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Match repair summary to a repair guide using vector database.
        
        Args:
            repair_summary: Repair summary text from scraped data
            fault_codes: Optional list of fault codes for filtering
            symptoms: Optional symptom text for LLM-guided verification
        
        Returns:
            Dict with matched repair guide info or None if no good match
        """
        if not repair_summary or not isinstance(repair_summary, str):
            return None
        
        repair_summary = repair_summary.strip()
        if not repair_summary:
            return None
        
        summary_excerpt = repair_summary[:120].replace("\n", " ")
        if self.verbose:
            logger.info(
                "Matching summary: excerpt=%r, fault_codes=%s",
                summary_excerpt,
                fault_codes or [],
            )
        encode_start = time.perf_counter()
        try:
            # Generate embedding for repair summary
            embedding = self.encoder.encode(repair_summary, normalize=True)
            
            # Convert to numpy array (remove batch dimension if single item)
            if embedding.dim() > 1:
                embedding = embedding.squeeze(0)
            embedding_np = embedding.detach().cpu().numpy()
            if self.verbose:
                logger.info("Encoded summary in %.2fs", time.perf_counter() - encode_start)
            
            search_start = time.perf_counter()
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
            if self.verbose:
                logger.info(
                    "Vector search returned %s candidates in %.2fs (filter=%s)",
                    len(results),
                    time.perf_counter() - search_start,
                    "with_filters" if filter_dict else "none",
                )
            
            best_score = results[0]['score']
            if best_score < self.min_similarity:
                # Sample scores for diagnostics (first 200)
                if len(self.stats['below_threshold_scores']) < 200:
                    self.stats['below_threshold_scores'].append(best_score)
                return None
            
            candidates = results[: min(self.top_k, len(results))]
            best_match: Optional[Dict[str, Any]] = None
            if self.verbose:
                logger.info(
                    "Top-%s candidates: %s",
                    len(candidates),
                    [
                        {
                            "title": c.get("title") or c.get("procedure_name", ""),
                            "score": round(c.get("score", 0.0), 4),
                        }
                        for c in candidates
                    ],
                )

            if self.llm_gating:
                self.stats['llm_gating_enabled'] += 1
                if not self._reasoning_generator:
                    self.stats['llm_gating_failures'] += 1
                    return None
                for candidate in candidates:
                    if self.verbose:
                        logger.info(
                            "Evaluating LLM gating for candidate: title=%r, score=%s",
                            candidate.get("title") or candidate.get("procedure_name", ""),
                            round(candidate.get("score", 0.0), 4),
                        )
                    reasoning = self._reasoning_generator.generate(
                        repair_summary=repair_summary,
                        guide_title=candidate.get('title') or candidate.get('procedure_name', ''),
                        guide_text=candidate.get('text', ''),
                        fault_codes=fault_codes or None,
                        symptoms=symptoms,
                    )
                    if not reasoning:
                        self.stats['llm_gating_failures'] += 1
                        continue
                    relevance = reasoning.get("relevance_score")
                    if relevance is None:
                        self.stats['llm_gating_failures'] += 1
                        continue
                    try:
                        relevance = float(relevance)
                    except (TypeError, ValueError):
                        self.stats['llm_gating_failures'] += 1
                        continue
                    candidate["llm_relevance_score"] = relevance
                    candidate["match_reasoning"] = reasoning
                    if relevance >= self.llm_min_confidence:
                        if (
                            best_match is None
                            or relevance > best_match.get("llm_relevance_score", 0.0)
                        ):
                            best_match = candidate
                if best_match is None:
                    self.stats['llm_gating_rejected'] += 1
                    if self.verbose:
                        logger.info("LLM gating rejected candidate set.")
                    return None
                self.stats['llm_gating_passed'] += 1
            else:
                best_match = candidates[0]
                if self.verbose:
                    logger.info(
                        "LLM gating disabled; using top candidate score=%s",
                        round(best_match.get("score", 0.0), 4),
                    )
                if self.generate_reasoning and self._reasoning_generator:
                    best_match["match_reasoning"] = self._reasoning_generator.generate(
                        repair_summary=repair_summary,
                        guide_title=best_match.get('title') or best_match.get('procedure_name', ''),
                        guide_text=best_match.get('text', ''),
                        fault_codes=fault_codes or None,
                        symptoms=symptoms,
                    )
                    if not best_match["match_reasoning"]:
                        best_match["match_reasoning"] = None

            if best_match is None:
                return None
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
                'llm_relevance_score': best_match.get('llm_relevance_score'),
                'match_reasoning': best_match.get('match_reasoning'),
                'all_candidates': [
                    {
                        'title': r.get('title') or r.get('procedure_name', ''),
                        'procedure_id': r.get('procedure_id', ''),
                        'score': r['score'],
                        'llm_relevance_score': r.get('llm_relevance_score'),
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
        match = self.match_repair_guide(
            repair_summary=repair_summary,
            fault_codes=fault_codes,
            symptoms=record.get('symptoms'),
        )
        
        if match:
            self.stats['matched'] += 1
            record['repair_guide'] = {
                'title': match['title'],
                'procedure_id': match['procedure_id'],
                'similarity_score': match['similarity_score'],
                'llm_relevance_score': match.get('llm_relevance_score'),
                'text_preview': match['text'],
                'candidates': match['all_candidates']
            }
            record['match_status'] = 'matched'
            record['match_reasoning'] = match.get("match_reasoning")
        else:
            self.stats['no_match'] += 1
            record['repair_guide'] = None
            record['match_status'] = 'no_match'
        
        return record
    
    def process_from_db(
        self,
        db_url: str,
        unmatched_only: bool = True,
        dry_run: bool = False,
        batch_size: int = 200,
        db_connect_retries: int = 4,
    ) -> None:
        """
        Load from scraped_records, match each record, UPDATE rows in place with
        matched_guide_id, matched_guide_title, match_reasoning.
        """
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import OperationalError

        # Transient Neon SSL/network failures are common in long-running jobs;
        # configure sane defaults and retry connection creation at startup.
        db_connect_retries = max(1, int(db_connect_retries))
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=2,
            max_overflow=4,
            connect_args={"connect_timeout": 20},
        )
        batch_size = max(1, batch_size)
        logger.info("Loading records from scraped_records in batches of %s...", batch_size)
        if dry_run:
            logger.info("Dry-run mode enabled: no updates will be written to DB.")

        where = "repair_summary IS NOT NULL AND source_url IS NOT NULL"
        if unmatched_only:
            where += " AND matched_guide_id IS NULL"

        def _connect_with_retry():
            last_error = None
            for attempt in range(1, db_connect_retries + 1):
                try:
                    if attempt > 1:
                        logger.info("Retrying DB connection... attempt %s/%s", attempt, db_connect_retries)
                    return engine.connect()
                except OperationalError as exc:
                    last_error = exc
                    if attempt >= db_connect_retries:
                        break
                    delay = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "DB connection attempt %s/%s failed: %s. Retrying in %.1fs",
                        attempt,
                        db_connect_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise last_error

        total_processed = 0
        batch_number = 0
        cursor_value = None

        try:
            with _connect_with_retry() as conn:
                while True:
                    batch_number += 1
                    cursor_clause = ""
                    params = {"batch_size": batch_size}
                    if cursor_value is not None:
                        cursor_clause = " AND source_url > :cursor"
                        params["cursor"] = cursor_value

                    query = f"""
                        SELECT source_url, repair_summary, fault_codes, symptoms
                        FROM scraped_records
                        WHERE {where}{cursor_clause}
                        ORDER BY source_url
                        LIMIT :batch_size
                    """
                    query_start = time.perf_counter()
                    logger.info(
                        "Running DB batch query #%s (source_url > %r, limit=%s)...",
                        batch_number,
                        cursor_value,
                        batch_size,
                    )
                    result = conn.execute(text(query), params)
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result]
                    query_elapsed = time.perf_counter() - query_start
                    if self.verbose:
                        logger.info(
                            "DB batch query #%s returned %s rows in %.2fs",
                            batch_number,
                            len(rows),
                            query_elapsed,
                        )
                    elif query_elapsed >= 2.0:
                        logger.info(
                            "DB batch query #%s returned %s rows in %.2fs",
                            batch_number,
                            len(rows),
                            query_elapsed,
                        )

                    if not rows:
                        if self.verbose:
                            logger.info(
                                "No rows returned for batch query window (cursor=%s). Finishing.",
                                cursor_value,
                            )
                        break

                    logger.info(
                        "Processing batch %s with %s records... (cursor=%s)",
                        batch_number,
                        len(rows),
                        cursor_value,
                    )
                    batch_started_at = time.perf_counter()

                    updates = []
                    for row_idx, row in enumerate(rows, start=1):
                        row_source_url = row["source_url"]
                        cursor_value = row_source_url
                        fault_codes = row.get("fault_codes")
                        if isinstance(fault_codes, str):
                            try:
                                fault_codes = json.loads(fault_codes) if fault_codes else []
                            except json.JSONDecodeError:
                                fault_codes = []
                        record = {
                            "source_url": row_source_url,
                            "repair_summary": row["repair_summary"],
                            "fault_codes": fault_codes if isinstance(fault_codes, list) else [],
                            "symptoms": row.get("symptoms"),
                        }
                        if row_idx == 1:
                            logger.info(
                                "First row received in batch %s: source_url=%r",
                                batch_number,
                                row_source_url,
                            )
                        if self.verbose and (row_idx == 1 or row_idx % self.log_every == 0):
                            logger.info(
                                "Processing row %s/%s in batch %s",
                                row_idx,
                                len(rows),
                                batch_number,
                            )
                        processed = self.process_record(record)
                        match_guide = processed.get("repair_guide")
                        match_reasoning = processed.get("match_reasoning")
                        matched_id = match_guide.get("procedure_id") if match_guide else None
                        matched_title = match_guide.get("title") if match_guide else None
                        reasoning_json = json.dumps(match_reasoning, ensure_ascii=False) if match_reasoning else None
                        total_processed += 1
                        updates.append(
                            {
                                "mid": matched_id,
                                "title": matched_title,
                                "reasoning": reasoning_json,
                                "url": row_source_url,
                            }
                        )
                        if total_processed % self.log_every == 0:
                            logger.info(
                                "Processed %s rows... current cursor=%s",
                                total_processed,
                                cursor_value,
                            )

                    if not dry_run:
                        with engine.begin() as tx:
                            tx.execute(
                                text("""
                                    UPDATE scraped_records
                                    SET matched_guide_id = :mid, matched_guide_title = :title,
                                        match_reasoning = CAST(:reasoning AS jsonb)
                                    WHERE source_url = :url
                                """),
                                updates,
                            )
                    if self.verbose:
                        logger.info("Wrote batch %s updates.", batch_number)

                    batch_elapsed = time.perf_counter() - batch_started_at
                    logger.info(
                        "Batch %s committed in %.2fs (records=%s).",
                        batch_number,
                        batch_elapsed,
                        len(updates),
                    )

                    if (total_processed % 1000) == 0:
                        logger.info(f"Processed {total_processed} records...")
        except OperationalError as exc:
            logger.error("Failed to connect to PostgreSQL/Neon after retries: %s", exc)
            raise
        if dry_run:
            logger.info("Dry-run complete. No rows were updated.")
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
        if self.stats.get('llm_gating_enabled'):
            logger.info("")
            logger.info("LLM gating:")
            logger.info(f"  Gated candidates evaluated: {self.stats['llm_gating_enabled']}")
            logger.info(f"  Gating passed: {self.stats['llm_gating_passed']}")
            logger.info(f"  Gating rejected: {self.stats['llm_gating_rejected']}")
            logger.info(f"  LLM evaluation failures: {self.stats['llm_gating_failures']}")
        
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
    parser.add_argument(
        '--llm-gating',
        action='store_true',
        help='Enable LLM relevance gating before persisting matched_guide_id.'
    )
    parser.add_argument(
        '--llm-min-confidence',
        type=float,
        default=0.7,
        help='Minimum LLM relevance score required when --llm-gating is set.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run matching and print stats without writing matched_guide_* columns.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        help='Process records in DB batches instead of loading all at once (default: 200).'
    )
    parser.add_argument(
        '--log-every',
        type=int,
        default=25,
        help='Log progress every N processed rows (default: 25).'
    )
    parser.add_argument(
        '--match-verbose',
        action='store_true',
        help='Enable detailed per-row matching logs (encoding, search, LLM gating, etc.).'
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
        llm_gating=args.llm_gating,
        llm_min_confidence=args.llm_min_confidence,
        log_every=args.log_every,
        verbose=args.match_verbose,
    )

    if use_db:
        if not db_url or not db_url.startswith("postgresql"):
            logger.error("DATABASE_URL required for DB mode. Set in .env or export.")
            sys.exit(1)
        matcher.process_from_db(
            db_url,
            unmatched_only=not args.all,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
        )
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
