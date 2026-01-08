# MIST Database Directory

This directory contains BMW ISTA diagnostic databases used by the MIST system.

## Primary Database

**DiagDocDb_DECRYPTED.sqlite** (or DiagDocDb_Decrypted.sqlite)
- **Purpose**: Main diagnostic database containing fault codes, ECUs, repair procedures, and diagnostic objects
- **Key Tables**:
  - `XEP_FAULTCODES`: Fault code definitions
  - `XEP_FAULTLABELS`: Fault code descriptions
  - `XEP_ECUVARIANTS`: ECU information
  - `XEP_INFOOBJECTS`: Repair procedures
  - `XEP_INFOSEGMENTS`: Procedure content
  - `RG_ECUFAULT_DOCIDS`: Fault-repair mappings
  - `XEP_REFDIAGOBJECTS`: Fault-diagnostic relationships
- **Usage**: Primary source for knowledge graph construction and repair guide retrieval

## XML Content Databases

**xmlvalueprimitive_ENGB.sqlite**
- **Purpose**: English XML content for repair procedures
- **Key Table**: `xmlvalueprimitive` (with FTS index `fts`)
- **Usage**: Provides actual text content for repair guides
- **Size**: Very large (GBs)

**xmlvalueprimitive_ENGB_complete.sqlite**
- **Purpose**: Complete version of English XML content
- **Usage**: Alternative/backup version

**xmlvalueprimitive_OTHER.sqlite**
- **Purpose**: Non-English XML content
- **Usage**: Multi-language support

## Stream Data Databases

**streamdataprimitive_OTHER.sqlite**
- **Purpose**: Binary blobs (images, graphics) for repair procedures
- **Key Table**: `streamdataprimitive`
- **Usage**: Provides images and graphics referenced in repair guides
- **Size**: Very large (GBs)

**streamdataprimitive_ENGB.sqlite**
- **Purpose**: English-specific binary content

**streamdataprimitive_OTHER_complete.sqlite**
- **Purpose**: Complete version of stream data

## Database Access

The MIST system uses the `paths.py` module to locate databases. It checks:
1. `mist/data/databases/` (new location)
2. `_databases/new databases/` (fallback)
3. `_databases/databases/` (fallback)

Environment variable `MIST_DATABASE_DIR` can override the default location.

## Migration

Databases were migrated from `_databases/` directories. To re-run migration:
```bash
python scripts/migrate_databases.py
```

## Notes

- These databases are very large (multiple GBs each)
- They are excluded from git via `.gitignore`
- Keep backups of these databases as they are essential for MIST operation
- The primary database (`DiagDocDb_DECRYPTED.sqlite`) is required for all operations
