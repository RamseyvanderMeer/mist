#!/usr/bin/env python3
"""
Pre-index readiness check for repair guide indexing.

Verifies all prerequisites before running index_repair_guides.py:
  1. ChromaDB credentials (CHROMA_DB_API_KEY, CHROMA_DB_TENANT)
  2. ISTA database (DiagDocDb_Decrypted.sqlite)
  3. XML content database (xmlvalueprimitive_ENGB.sqlite) - optional
  4. Config files (embedding_config.yaml, retrieval_config.yaml)
  5. ChromaDB connectivity
  6. ISTA database has procedures

Usage:
  python scripts/check_index_readiness.py

Exit code: 0 if ready, 1 if not ready.
"""
import os
import sys
from pathlib import Path

# Add project root for consistent imports (from src.X)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _redact(s: str, show_chars: int = 4) -> str:
    """Redact string for safe logging."""
    if not s or len(s) <= show_chars:
        return "***"
    return s[:show_chars] + "***" + s[-2:] if len(s) > show_chars + 2 else "***"


def check_env() -> tuple[bool, list[str]]:
    """Check required environment variables."""
    errors = []
    api_key = os.getenv("CHROMA_DB_API_KEY")
    tenant = os.getenv("CHROMA_DB_TENANT")

    if not api_key or not api_key.strip():
        errors.append("CHROMA_DB_API_KEY is not set in .env")
    if not tenant or not tenant.strip():
        errors.append("CHROMA_DB_TENANT is not set in .env")

    if errors:
        return False, errors
    return True, [f"CHROMA_DB_API_KEY: {_redact(api_key)}", f"CHROMA_DB_TENANT: {tenant[:8]}***"]


def check_configs(paths) -> tuple[bool, list[str]]:
    """Check config files exist and have required keys."""
    errors = []
    msgs = []

    if not paths.embedding_config.exists():
        errors.append(f"Embedding config not found: {paths.embedding_config}")
    else:
        import yaml
        with open(paths.embedding_config, "r") as f:
            ec = yaml.safe_load(f)
        fc = (ec or {}).get("models", {}).get("fault_code", {})
        if not fc.get("model_name"):
            errors.append("embedding_config.yaml: models.fault_code.model_name missing")
        else:
            msgs.append(f"Embedding model: {fc.get('model_name')}")

    if not paths.retrieval_config.exists():
        errors.append(f"Retrieval config not found: {paths.retrieval_config}")
    else:
        import yaml
        with open(paths.retrieval_config, "r") as f:
            rc = yaml.safe_load(f)
        vs = (rc or {}).get("vector_store", {})
        if vs.get("provider") != "chromadb":
            errors.append("retrieval_config.yaml: vector_store.provider should be 'chromadb'")
        msgs.append(f"Collection: {vs.get('collection_name', 'repair_guides_enhanced')}")

    return len(errors) == 0, errors if errors else msgs


def check_ista_db(paths) -> tuple[bool, list[str]]:
    """Check ISTA database exists and has procedures."""
    errors = []
    for name in ("DiagDocDb_Decrypted.sqlite", "DiagDocDb_DECRYPTED.sqlite"):
        p = paths.get_database_path(name)
        if p.exists():
            try:
                from sqlalchemy import create_engine, text
                engine = create_engine(f"sqlite:///{p}")
                with engine.connect() as conn:
                    r = conn.execute(text("SELECT COUNT(*) FROM XEP_INFOOBJECTS WHERE ID IS NOT NULL"))
                    count = r.scalar() or 0
                if count == 0:
                    errors.append(f"ISTA DB {name} exists but has 0 procedures (wrong or empty DB?)")
                else:
                    return True, [f"ISTA DB: {name} ({count:,} procedures)"]
            except Exception as e:
                errors.append(f"ISTA DB {name}: {e}")
            break
    else:
        p = paths.get_database_path("DiagDocDb_Decrypted.sqlite")
        errors.append(f"ISTA database not found. Expected: {p}")
        errors.append("  Obtain DiagDocDb_Decrypted.sqlite from BMW ISTA and place in data/databases/")

    return False, errors


def check_xml_db(paths) -> tuple[bool, list[str]]:
    """Check XML content database (optional)."""
    for name in ("xmlvalueprimitive_ENGB.sqlite", "xmlvalueprimitive_ENGB_complete.sqlite"):
        p = paths.get_database_path(name)
        if p.exists():
            return True, [f"XML DB: {name} (full procedure content available)"]
    return True, ["XML DB: not found (indexing will use titles only, less semantic richness)"]


def check_chromadb(paths) -> tuple[bool, list[str]]:
    """Test ChromaDB connectivity."""
    try:
        import yaml
        with open(paths.retrieval_config, "r") as f:
            rc = yaml.safe_load(f)
        vs_config = (rc or {}).get("vector_store", {})
        from src.retrieval.vector_store import VectorStore
        store = VectorStore(vs_config)
        # Quick operation to verify connection
        _ = store.get_collection_info()
        return True, ["ChromaDB: connected successfully"]
    except Exception as e:
        return False, [f"ChromaDB: {e}"]


def main() -> int:
    print("=" * 60)
    print("Index Readiness Check")
    print("=" * 60)

    try:
        from src.paths import get_paths
        paths = get_paths()
    except Exception as e:
        print(f"FAIL: Could not load paths: {e}")
        return 1

    all_ok = True
    checks = [
        ("Environment (ChromaDB)", check_env),
        ("Config files", lambda: check_configs(paths)),
        ("ISTA database", lambda: check_ista_db(paths)),
        ("XML database (optional)", lambda: check_xml_db(paths)),
        ("ChromaDB connectivity", lambda: check_chromadb(paths)),
    ]

    for name, fn in checks:
        ok, msgs = fn()
        status = "OK" if ok else "FAIL"
        symbol = "[OK]" if ok else "[X]"
        print(f"\n{symbol} {name}: {status}")
        for m in msgs:
            print(f"    {m}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("READY: You can run: python scripts/index_repair_guides.py")
        return 0
    else:
        print("NOT READY: Fix the issues above before indexing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
