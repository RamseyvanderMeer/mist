#!/usr/bin/env python3
"""
End-to-end test: pairs of scraped records + ISTA procedures with overlapping info.

1. Discover valid pairs: scraped records with fault_codes that have matching
   ISTA procedures (via get_procedures_for_fault) and substantive repair summaries
2. Index multiple procedures (expected + extras) so retrieval is non-trivial
3. Run retrieval for each pair and verify expected procedure ranks #1
4. Log everything for verification

Usage:
  python scripts/e2e_one_procedure_one_record.py [--main-collection] [--no-cleanup] [--pairs N]

  --main-collection   Use main repair_guides collection (default: e2e test collection)
  --no-cleanup        Do not delete test collection after run
  --pairs N           Max pairs to discover and test (default: 3)

Requires: DATABASE_URL (scraped_records), ISTA DB, CHROMA_DB_*, .env
"""
import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # .env may already be loaded

# Configure verbose logging - log everything
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Reduce noise from some libs
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

E2E_COLLECTION = "repair_guides_e2e_one_proc"
MAIN_COLLECTION = "repair_guides_enhanced"  # from retrieval_config.yaml


# Phrases that indicate generic/non-substantive repair summaries
GENERIC_PHRASES = (
    "repaired better and possibly cheaper",
    "repaired at a local shop",
    "fixed it",
    "problem solved",
    "no further information",
    "see above",
    "same as above",
)


def log_section(title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def _is_substantive_summary(repair_summary: Optional[str]) -> bool:
    """True if repair_summary is substantive (not generic forum filler)."""
    if not repair_summary or len(repair_summary.strip()) < 40:
        return False
    lower = repair_summary.lower().strip()
    return not any(phrase in lower for phrase in GENERIC_PHRASES)


def _parse_fault_codes(rec: Dict[str, Any]) -> List[str]:
    """Parse fault_codes from record (JSONB or list)."""
    fc = rec.get("fault_codes")
    if isinstance(fc, str):
        try:
            fc = json.loads(fc) if fc else []
        except json.JSONDecodeError:
            fc = []
    return [str(c).strip() for c in (fc or []) if c and str(c).strip()]


def discover_valid_pairs(
    db_url: str,
    ista_db,
    max_pairs: int = 3,
) -> List[tuple]:
    """
    Find (scraped_record, procedure) pairs with overlapping fault codes and substantive summaries.

    Returns list of (scraped_record, procedure) where:
    - scraped_record has non-empty fault_codes
    - procedure is linked via get_procedures_for_fault
    - repair_summary is substantive (not generic)
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    seen_proc_ids: set = set()
    pairs: List[tuple] = []

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT source_url, fault_codes, repair_summary, symptoms
                FROM scraped_records
                WHERE repair_summary IS NOT NULL AND repair_summary != ''
                  AND fault_codes IS NOT NULL AND fault_codes != '[]' AND fault_codes != '{}'
                ORDER BY created_at DESC
                LIMIT 500
            """)
        )
        rows = result.fetchall()
        cols = result.keys()

    for row in rows:
        if len(pairs) >= max_pairs:
            break
        rec = dict(zip(cols, row))
        rec["fault_codes"] = _parse_fault_codes(rec)
        if not rec["fault_codes"]:
            continue
        if not _is_substantive_summary(rec.get("repair_summary")):
            continue

        for code in rec["fault_codes"][:5]:
            try:
                procs = ista_db.get_procedures_for_fault(str(code).strip())
                for p in procs[:1]:  # first matching procedure per code
                    proc_id = str(p.get("ID", p.get("id", "")))
                    if proc_id in seen_proc_ids:
                        continue
                    seen_proc_ids.add(proc_id)
                    procedure = {
                        "id": proc_id,
                        "title_engb": str(p.get("TITLE_ENGB", p.get("title_engb", "")) or ""),
                        "name": str(p.get("NAME", p.get("name", "")) or ""),
                    }
                    pairs.append((rec, procedure))
                    break  # one pair per record
            except Exception as e:
                logger.debug("get_procedures_for_fault %s: %s", code, e)
                continue
            if len(pairs) >= max_pairs:
                break

    return pairs


def fetch_one_scraped_record(db_url: str) -> Optional[Dict[str, Any]]:
    """Fetch one scraped record with repair_summary and fault_codes."""
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT source_url, fault_codes, repair_summary, symptoms
                FROM scraped_records
                WHERE repair_summary IS NOT NULL AND repair_summary != ''
                LIMIT 1
            """)
        )
        row = result.fetchone()
    if not row:
        return None
    cols = result.keys()
    rec = dict(zip(cols, row))
    fault_codes = rec.get("fault_codes")
    if isinstance(fault_codes, str):
        try:
            fault_codes = json.loads(fault_codes) if fault_codes else []
        except json.JSONDecodeError:
            fault_codes = []
    rec["fault_codes"] = fault_codes if isinstance(fault_codes, list) else []
    return rec


def find_procedure_for_record(ista_db, scraped_record: Dict) -> Optional[Dict]:
    """Find a procedure that matches the scraped record (by fault code or any)."""
    fault_codes = scraped_record.get("fault_codes") or []
    for code in fault_codes[:5]:  # Try first 5 codes
        if not code or not str(code).strip():
            continue
        try:
            procs = ista_db.get_procedures_for_fault(str(code).strip())
            if procs:
                p = procs[0]
                return {
                    "id": str(p.get("ID", p.get("id", ""))),
                    "title_engb": str(p.get("TITLE_ENGB", p.get("title_engb", "")) or ""),
                    "name": str(p.get("NAME", p.get("name", "")) or ""),
                }
        except Exception as e:
            logger.debug("get_procedures_for_fault %s: %s", code, e)
            continue

    # Fallback: any procedure with limit 1
    from sqlalchemy import text
    with ista_db.connection.session() as session:
        result = session.execute(
            text("SELECT ID, TITLE_ENGB, NAME FROM XEP_INFOOBJECTS WHERE ID IS NOT NULL LIMIT 1")
        )
        row = result.fetchone()
    if row:
        return {
            "id": str(row[0]),
            "title_engb": str(row[1] or ""),
            "name": str(row[2] or ""),
        }
    return None


def _index_procedure(
    procedure: Dict[str, Any],
    ista_db,
    xml_fetcher,
    encoder,
    vector_store,
    collection_name: str,
) -> tuple:
    """Index one procedure. Returns (doc_emb_np, doc, fault_codes)."""
    import numpy as np
    import torch

    fault_codes = ista_db.get_fault_codes_for_procedure(procedure["id"])
    fault_labels = ista_db.get_fault_labels_for_procedure(procedure["id"])
    title = procedure.get("title_engb") or procedure.get("name", "")
    xml_content = xml_fetcher.get_content(procedure["id"], title)
    text_content = (title + "\n\n" + (xml_content or "")) if xml_content else title

    prefix_parts = []
    if fault_codes:
        prefix_parts.append(f"Fault codes: {', '.join(fault_codes)}.")
    if fault_labels:
        prefix_parts.append(f"Problem descriptions: {'; '.join(fault_labels[:5])}.")
    fault_prefix = "\n\n".join(prefix_parts) + "\n\n" if prefix_parts else ""
    text_for_embedding = fault_prefix + text_content

    with torch.no_grad():
        doc_emb = encoder.encode(text_for_embedding, normalize=True, is_query=False)
    doc_emb_np = doc_emb.cpu().numpy()
    if doc_emb_np.ndim > 1:
        doc_emb_np = doc_emb_np.squeeze(0)

    doc = {
        "id": procedure["id"],
        "text": text_for_embedding[:50000],
        "title": title,
        "procedure_id": procedure["id"],
        "procedure_name": procedure.get("name", ""),
        "fault_codes": fault_codes,
        "ecu_category": "",
    }
    vector_store.add(np.expand_dims(doc_emb_np, 0), [doc], batch_size=1)
    return doc_emb_np, doc, fault_codes


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E test: pairs with overlapping fault codes")
    parser.add_argument("--main-collection", action="store_true", help="Use main collection")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not delete test collection")
    parser.add_argument("--pairs", type=int, default=3, help="Max pairs to discover and test")
    args = parser.parse_args()

    use_main = args.main_collection
    collection_name = MAIN_COLLECTION if use_main else E2E_COLLECTION

    log_section("E2E: Discover Pairs -> Index -> Retrieve -> Verify")

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set. Cannot fetch scraped_records.")
        return 1

    # Paths and config
    from paths import get_paths
    paths = get_paths()
    import yaml
    import numpy as np
    import torch

    with open(paths.embedding_config, "r", encoding="utf-8") as f:
        embedding_config = yaml.safe_load(f)
    with open(paths.retrieval_config, "r", encoding="utf-8") as f:
        retrieval_config = yaml.safe_load(f)

    vs_config = dict(retrieval_config.get("vector_store", {}))
    vs_config["collection_name"] = collection_name
    logger.info("Using collection: %s", collection_name)

    # For e2e test collection: delete if exists to avoid "collection soft deleted" from prior run
    if not use_main:
        try:
            import chromadb
            _api_key = os.getenv("CHROMA_DB_API_KEY")
            _tenant = os.getenv("CHROMA_DB_TENANT")
            _db = vs_config.get("database", "mist")
            if _api_key and _tenant:
                _client = chromadb.CloudClient(api_key=_api_key, tenant=_tenant, database=_db)
                _client.delete_collection(collection_name)
                logger.info("Cleared prior test collection for fresh run")
        except Exception as e:
            logger.debug("No prior collection to clear (or already gone): %s", e)

    from database.ista_db import IstaDatabase
    from database.xml_content import XmlContentFetcher
    from embeddings.fault_code_encoder import FaultCodeEncoder
    from retrieval.vector_store import VectorStore
    from sqlalchemy import text

    ista_db = IstaDatabase()
    fc_config = embedding_config.get("models", {}).get("fault_code", {})
    encoder = FaultCodeEncoder(
        model_name=fc_config.get("model_name", "intfloat/e5-mistral-7b-instruct"),
        device=fc_config.get("device", "cpu"),
        projection_dim=fc_config.get("projection_dim", 1024),
    )
    xml_fetcher = XmlContentFetcher()
    vector_store = VectorStore(vs_config)
    logger.info("Encoder: dim=%s", encoder.get_dimension())

    # --- 1. Discover valid pairs (fault code overlap + substantive summaries) ---
    log_section("1. Discover valid pairs (fault codes + substantive repair_summary)")
    pairs = discover_valid_pairs(db_url, ista_db, max_pairs=args.pairs)

    if not pairs:
        logger.warning("No valid pairs found. Falling back to any record + any procedure.")
        scraped = fetch_one_scraped_record(db_url)
        if not scraped:
            logger.error("No scraped record found with repair_summary.")
            return 1
        procedure = find_procedure_for_record(ista_db, scraped)
        if not procedure:
            logger.error("No procedure found in ISTA DB.")
            return 1
        pairs = [(scraped, procedure)]

    for i, (rec, proc) in enumerate(pairs):
        logger.info(
            "  Pair %d: fault_codes=%s -> procedure id=%s title=%s",
            i + 1,
            rec.get("fault_codes", []),
            proc["id"],
            (proc.get("title_engb", "") or "")[:50],
        )

    # --- 2. Collect procedures to index (expected + extras for non-trivial retrieval) ---
    log_section("2. Collect procedures to index")
    expected_proc_ids = {p["id"] for _, p in pairs}
    procedures_to_index = list({p["id"]: p for _, p in pairs}.values())  # unique by id

    # Add 2-4 extra procedures so retrieval is not trivial
    extra_needed = max(0, 5 - len(procedures_to_index))
    if extra_needed > 0:
        with ista_db.connection.session() as session:
            result = session.execute(
                text("SELECT ID, TITLE_ENGB, NAME FROM XEP_INFOOBJECTS WHERE ID IS NOT NULL LIMIT 100")
            )
            for row in result.fetchall():
                if len(procedures_to_index) >= 5:
                    break
                pid = str(row[0])
                if pid in expected_proc_ids:
                    continue
                procedures_to_index.append({
                    "id": pid,
                    "title_engb": str(row[1] or ""),
                    "name": str(row[2] or ""),
                })

    logger.info("Indexing %d procedures (%d expected matches)", len(procedures_to_index), len(expected_proc_ids))

    # --- 3. Index all procedures ---
    log_section("3. Index procedures to vector store")
    for proc in procedures_to_index:
        _index_procedure(proc, ista_db, xml_fetcher, encoder, vector_store, collection_name)
        logger.info("  Indexed: id=%s title=%s", proc["id"], (proc.get("title_engb", "") or "")[:50])

    # --- 4. Run retrieval for each pair and verify ---
    log_section("4. Run retrieval and verify each pair")
    threshold = 0.5
    all_passed = True

    for i, (scraped, procedure) in enumerate(pairs):
        query_fault_codes = scraped.get("fault_codes") or []
        description = scraped.get("repair_summary") or scraped.get("symptoms") or ""
        if description and scraped.get("symptoms"):
            description = f"{scraped.get('symptoms', '')}. {description}".strip()
        query_text = f"Fault codes: {', '.join(query_fault_codes)}. Problem: {description}" if query_fault_codes else f"Problem: {description}"
        if not query_text.strip():
            query_text = description or "repair"

        logger.info("Pair %d query: fault_codes=%s desc=%s", i + 1, query_fault_codes, (description or "")[:80])

        with torch.no_grad():
            query_emb = encoder.encode(query_text, normalize=True, is_query=True)
        query_emb_np = query_emb.cpu().numpy()
        if query_emb_np.ndim > 1:
            query_emb_np = query_emb_np.flatten()

        results = vector_store.search(query_embedding=query_emb_np, top_k=10, filter_dict=None)
        our_proc_id = procedure["id"]
        found = [r for r in results if r.get("procedure_id") == our_proc_id]

        if found:
            idx = results.index(found[0])
            score = found[0]["score"]
            passed = score >= threshold
            if passed:
                logger.info("  Pair %d PASS: procedure %s rank=%d score=%.4f", i + 1, our_proc_id, idx + 1, score)
            else:
                logger.warning("  Pair %d LOW: procedure %s rank=%d score=%.4f < %.2f", i + 1, our_proc_id, idx + 1, score, threshold)
                all_passed = False
        else:
            logger.warning("  Pair %d FAIL: procedure %s not in top-%d", i + 1, our_proc_id, len(results))
            all_passed = False

        for j, r in enumerate(results[:5]):
            logger.debug("    [%d] score=%.4f procedure_id=%s", j + 1, r["score"], r.get("procedure_id", ""))

    log_section("Summary")
    logger.info("Pairs tested: %d", len(pairs))
    logger.info("Procedures indexed: %d", len(procedures_to_index))
    logger.info("Collection: %s", collection_name)
    if not use_main and not args.no_cleanup:
        try:
            vector_store.client.delete_collection(collection_name)
            logger.info("Cleaned up test collection %s", collection_name)
        except Exception as e:
            logger.warning("Could not delete test collection: %s", e)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
