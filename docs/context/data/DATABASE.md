# MIST Database Documentation

This document consolidates information about BMW ISTA databases used by MIST.

## Overview

MIST uses BMW ISTA diagnostic databases to build knowledge graphs and retrieve repair guides. The databases contain fault codes, ECUs, diagnostic procedures, and repair instructions.

## Database Files

See [data/databases/README.md](../../../data/databases/README.md) for detailed information about each database file.

## Access Methods

### Using Paths Module

The recommended way to access databases is through the `paths.py` module:

```python
from src.paths import get_paths

paths = get_paths()
db_path = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
```

### Direct Access

```python
import sqlite3

db_path = "mist/data/databases/DiagDocDb_DECRYPTED.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query example
cursor.execute("SELECT CODE, TITLE_ENGB FROM XEP_FAULTCODES LIMIT 10")
results = cursor.fetchall()
```

## Key Tables

### Fault Codes
- `XEP_FAULTCODES`: Fault code definitions
- `XEP_FAULTLABELS`: Fault code descriptions
- `XEP_COMBINEDFAULTS`: Combined fault information

### ECUs
- `XEP_ECUVARIANTS`: ECU variants
- `XEP_ECUGROUPS`: ECU groups

### Repair Procedures
- `XEP_INFOOBJECTS`: Repair procedure metadata
- `XEP_INFOSEGMENTS`: Procedure content segments
- `RG_ECUFAULT_DOCIDS`: Mapping between faults and repair procedures

### Diagnostic Objects
- `XEP_DIAGNOSISOBJECTS`: Diagnostic test procedures
- `XEP_REFDIAGOBJECTS`: Fault-diagnostic relationships
- `XEP_REFDIAGNOSISTREE`: Diagnostic tree structure

### BMW Fault Mappings (mist_data.db)

Populated by `mist-cli fetch-bmwfault`:

- **bmwfault_pcodes**: Full rows from bmwfault.codes Lookup (PCode, Code, Label, ECU Variant, ECU Label). `fault_info` column stores JSON from DiagView pages (fault_code_description, service_plan, fault_impact, service_notes, etc.).
- **bmwfault_mappings**: Aggregated pcode → comma-separated hex codes for OBD→ISTA lookup in `get_lookup_variants()`.

Retrieval uses these mappings to align user P-codes (e.g. P0301) with ISTA hex codes (e.g. 29CC) in vector search, KG lookups, and `ista_db.get_procedures_for_fault()`. Run `mist-cli fetch-bmwfault` to improve P-code coverage.

### Retrieved Records (Neon/Postgres)

- **retrieved_records**: Stores retrieval evaluation results (ground truth vs. retrieved guides). Populated by `tests/test_retrieval_evaluation.py` when `RETRIEVAL_EVAL_PERSIST=1`. Run migration: `python scripts/run_retrieved_records_migration.py`.

## Database Schema

For document hierarchy and Process Analysis (preliminary tasks, Ref:h3 links), see [ISTA_DATABASE_GUIDE.md](ISTA_DATABASE_GUIDE.md).

## Integration with MIST

MIST uses databases for:
1. **Knowledge Graph Construction**: Extracts relationships between faults, ECUs, and procedures
2. **Repair Guide Indexing**: Indexes repair procedures in vector store
3. **Query Processing**: Retrieves relevant procedures based on fault codes

## Environment Configuration

Set `ISTA_DB_PATH` environment variable to override default database location:
```bash
export ISTA_DB_PATH=./data/databases/DiagDocDb_DECRYPTED.sqlite
```

Or use `MIST_DATABASE_DIR` to override the entire database directory:
```bash
export MIST_DATABASE_DIR=./data/databases
```
