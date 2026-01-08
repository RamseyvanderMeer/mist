# MIST Migration Guide

This guide documents what was moved and how to update existing scripts.

## What Was Moved

### Databases
All SQLite databases were copied from:
- `_databases/new databases/`
- `_databases/databases/`
- `_databases/` (root level)

To:
- `mist/data/databases/`

**Primary Database**: `DiagDocDb_DECRYPTED.sqlite` (or `DiagDocDb_Decrypted.sqlite`)

### Documentation
- `_docs/DATABASE_ACCESS_GUIDE.md` → `mist/docs/DATABASE_ACCESS_GUIDE.md`
- `_docs/ISTA_DATABASE_GUIDE.md` → `mist/docs/ISTA_DATABASE_GUIDE.md`
- `mist/implementation.md` → `mist/docs/legacy/implementation.md`
- `mist/knoulage transfer.md` → `mist/docs/legacy/knowledge_transfer.md`

New consolidated docs:
- `mist/docs/DATABASE.md`: Consolidated database documentation
- `mist/docs/ARCHITECTURE.md`: Architecture overview

## Updating Existing Scripts

### Option 1: Use Path Management (Recommended)

Update scripts to use the centralized path management:

```python
from mist.src.paths import get_paths

paths = get_paths()
db_path = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
```

This automatically checks:
1. `mist/data/databases/` (new location)
2. `_databases/new databases/` (fallback)
3. `_databases/databases/` (fallback)

### Option 2: Update Direct Paths

Update hardcoded paths in scripts:

**Old:**
```python
DB_PATH = "_databases/new databases/DiagDocDb_DECRYPTED.sqlite"
```

**New:**
```python
DB_PATH = "mist/data/databases/DiagDocDb_DECRYPTED.sqlite"
```

Or use environment variable:
```python
import os
DB_PATH = os.getenv("ISTA_DB_PATH", "mist/data/databases/DiagDocDb_DECRYPTED.sqlite")
```

### Option 3: Environment Variable Override

Set environment variable to override default location:
```bash
export ISTA_DB_PATH=./mist/data/databases/DiagDocDb_DECRYPTED.sqlite
```

Or override entire database directory:
```bash
export MIST_DATABASE_DIR=./mist/data/databases
```

## Backward Compatibility

The path management system (`src/paths.py`) includes fallback mechanisms:
- Checks new location first (`mist/data/databases/`)
- Falls back to old locations if not found
- Logs warnings when using fallback paths

This ensures existing scripts continue working during transition.

## Migration Script

To re-run database migration:
```bash
python mist/scripts/migrate_databases.py
```

This script:
- Copies databases from old locations to new location
- Skips files that already exist and have same size
- Reports progress and errors

## Verification

Verify migration completed successfully:

1. **Check Database Location**
   ```python
   from mist.src.paths import get_paths
   paths = get_paths()
   db_path = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
   print(f"Database path: {db_path}")
   print(f"Exists: {db_path.exists()}")
   ```

2. **Test Database Access**
   ```python
   import sqlite3
   from mist.src.paths import get_paths
   
   paths = get_paths()
   db_path = paths.get_database_path("DiagDocDb_DECRYPTED.sqlite")
   conn = sqlite3.connect(str(db_path))
   cursor = conn.cursor()
   cursor.execute("SELECT COUNT(*) FROM XEP_FAULTCODES")
   count = cursor.fetchone()[0]
   print(f"Fault codes in database: {count}")
   conn.close()
   ```

3. **Check File Sizes**
   ```bash
   ls -lh mist/data/databases/*.sqlite
   ```

## Troubleshooting

### Database Not Found
- Check that migration script ran successfully
- Verify database files exist in `mist/data/databases/`
- Check environment variables if using overrides

### Path Resolution Issues
- Ensure `src/paths.py` is in Python path
- Check that MIST root directory is correct
- Verify fallback paths exist

### Import Errors
- Add `mist/src` to Python path:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent.parent / "mist" / "src"))
  ```

## Next Steps

1. Update scripts to use path management system
2. Test all database access points
3. Remove old database references once verified
4. Update documentation with new paths
