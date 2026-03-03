#!/usr/bin/env python3
"""
Extract valid repair guide titles from MIST database and vector store.

This script extracts all repair guide titles from:
1. BMW ISTA database (XEP_INFOOBJECTS table)
2. Vector store (ChromaDB collection)
3. Outputs a list of valid titles for web scraping agents to use for matching

The output can be used by scraping agents to match scraped content against
known valid repair guide titles using fuzzy matching.
"""
import sys
import json
import csv
import sqlite3
import re
import time
import signal
import atexit
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
import logging
from collections import defaultdict

# Add project root for consistent imports (from src.X)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.database.ista_db import IstaDatabase
from src.retrieval.vector_store import VectorStore
from src.paths import get_paths
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepairGuideTitleExtractor:
    """Extract repair guide titles from MIST database and vector store."""
    
    def __init__(
        self,
        include_descriptions: bool = False,
        max_description_length: int = 500,
        use_xml_db: bool = True,
        use_llm: bool = False,
        llm_provider: Optional[str] = None,
        incremental_file: Optional[Path] = None
    ):
        """
        Initialize extractor.
        
        Args:
            include_descriptions: If True, include procedure descriptions/summaries
            max_description_length: Maximum length of description summary (chars)
            use_xml_db: If True, try to extract descriptions from xmlvalueprimitive database
            use_llm: If True, use LLM to generate summaries for missing descriptions
            llm_provider: LLM provider to use ('openai', 'anthropic', 'open_source', 'gemini')
            incremental_file: Path to existing CSV file to load for incremental processing
        """
        self.titles: Set[str] = set()
        self.title_to_procedure: Dict[str, Dict[str, Any]] = {}
        self.include_descriptions = include_descriptions
        self.max_description_length = max_description_length
        self.use_xml_db = use_xml_db
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.xml_db_connection = None
        self.llm_client = None
        self.llm_disabled = False  # Track if LLM was disabled due to errors
        self.llm_failure_count = 0  # Track consecutive LLM failures
        self.llm_max_failures = 3  # Disable LLM after 3 consecutive failures
        self.stats = {
            'ista_titles': 0,
            'vector_store_titles': 0,
            'duplicates': 0,
            'total_unique': 0,
            'with_descriptions': 0,
            'from_xml_db': 0,
            'from_llm': 0,
            'llm_errors': 0,
            'loaded_from_existing': 0,
            'skipped_with_description': 0,
            'new_titles': 0,
            'new_descriptions': 0
        }
        # Checkpointing
        self.checkpoint_path: Optional[Path] = None
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 30  # Save every 30 seconds
        self.checkpoint_row_interval = 1000  # Save every 1000 rows
        self.checkpoint_enabled = True
        self._shutdown_requested = False
        
        # Load existing data if incremental mode
        if incremental_file and incremental_file.exists():
            self._load_existing_csv(incremental_file)
    
    def _init_xml_db(self) -> Optional[sqlite3.Connection]:
        """Initialize connection to xmlvalueprimitive database."""
        if not self.use_xml_db:
            logger.info("XML database extraction disabled (--no-xml-db or use_xml_db=False)")
            return None
        
        try:
            paths = get_paths()
            # Try to find xmlvalueprimitive database
            db_paths = [
                paths.get_database_path("xmlvalueprimitive_ENGB.sqlite"),
                paths.get_database_path("xmlvalueprimitive_ENGB_complete.sqlite"),
            ]
            
            logger.info(f"Searching for XML database in {len(db_paths)} possible locations...")
            for db_path in db_paths:
                if db_path:
                    logger.debug(f"  Checking: {db_path} (exists: {db_path.exists() if db_path else False})")
                    if db_path.exists():
                        logger.info(f"✓ Found XML database: {db_path}")
                        logger.info(f"  Database size: {db_path.stat().st_size / (1024*1024*1024):.2f} GB")
                        conn = sqlite3.connect(str(db_path))
                        conn.row_factory = sqlite3.Row
                        
                        # Verify table exists
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = [row[0] for row in cursor.fetchall()]
                        logger.info(f"  Found {len(tables)} tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}")
                        
                        if 'xmlvalueprimitive' in tables:
                            # Get row count
                            cursor.execute("SELECT COUNT(*) FROM xmlvalueprimitive")
                            count = cursor.fetchone()[0]
                            logger.info(f"  xmlvalueprimitive table has {count:,} rows")
                        
                        return conn
                    else:
                        logger.debug(f"  ✗ Not found: {db_path}")
            
            logger.warning("✗ XML database not found in any expected location, skipping XML content extraction")
            logger.warning("  Searched paths:")
            for db_path in db_paths:
                logger.warning(f"    - {db_path}")
            return None
        except Exception as e:
            logger.error(f"✗ Could not connect to XML database: {e}", exc_info=True)
            return None
    
    def _init_llm(self) -> Optional[Any]:
        """Initialize LLM client for generating descriptions."""
        if not self.use_llm:
            logger.info("LLM generation disabled (--use-llm not specified)")
            return None
        
        try:
            logger.info(f"Initializing LLM client (provider: {self.llm_provider})...")
            
            # Check environment variables
            import os
            if self.llm_provider == 'gemini':
                api_key = os.getenv('GEMINI_API_KEY')
                model = os.getenv('GEMINI_MODEL')
                logger.info(f"  GEMINI_API_KEY: {'✓ Found' if api_key else '✗ Not found'}")
                logger.info(f"  GEMINI_MODEL: {model if model else 'Not set (will use default)'}")
            elif self.llm_provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
                logger.info(f"  OPENAI_API_KEY: {'✓ Found' if api_key else '✗ Not found'}")
            elif self.llm_provider == 'anthropic':
                api_key = os.getenv('ANTHROPIC_API_KEY')
                logger.info(f"  ANTHROPIC_API_KEY: {'✓ Found' if api_key else '✗ Not found'}")
            
            # Load LLM config
            paths = get_paths()
            logger.debug(f"Loading LLM config from: {paths.llm_config}")
            with open(paths.llm_config, 'r', encoding='utf-8') as f:
                llm_config = yaml.safe_load(f)
            
            provider = self.llm_provider or 'openai'
            logger.info(f"  Using provider: {provider}")
            
            if provider == 'openai':
                try:
                    from src.llm.openai_client import OpenAIClient
                    config = llm_config.get('openai', {})
                    logger.info(f"  Model: {config.get('model', 'default')}")
                    client = OpenAIClient(config)
                    logger.info("✓ OpenAI client initialized successfully")
                    return client
                except (ImportError, ValueError) as e:
                    logger.error(f"✗ OpenAI client not available: {e}", exc_info=True)
            elif provider == 'anthropic':
                try:
                    from src.llm.anthropic_client import AnthropicClient
                    config = llm_config.get('anthropic', {})
                    logger.info(f"  Model: {config.get('model', 'default')}")
                    client = AnthropicClient(config)
                    logger.info("✓ Anthropic client initialized successfully")
                    return client
                except (ImportError, ValueError) as e:
                    logger.error(f"✗ Anthropic client not available: {e}", exc_info=True)
            elif provider == 'open_source':
                try:
                    from src.llm.open_source_client import OpenSourceClient
                    config = llm_config.get('open_source', {})
                    logger.info(f"  Model: {config.get('model', 'default')}")
                    client = OpenSourceClient(config)
                    logger.info("✓ Open source client initialized successfully")
                    return client
                except (ImportError, ValueError) as e:
                    logger.error(f"✗ Open source client not available: {e}", exc_info=True)
            elif provider == 'gemini':
                try:
                    from src.llm.gemini_client import GeminiClient
                    config = llm_config.get('gemini', {})
                    # Override model from environment variable if present
                    model = os.getenv('GEMINI_MODEL') or config.get('model', 'gemini-pro')
                    config['model'] = model  # Update config with the resolved model
                    logger.info(f"  Model: {model}")
                    client = GeminiClient(config)
                    logger.info("✓ Gemini client initialized successfully")
                    return client
                except (ImportError, ValueError) as e:
                    logger.error(f"✗ Gemini client not available: {e}", exc_info=True)
            
            logger.warning(f"✗ LLM provider '{provider}' not available, skipping LLM-based descriptions")
            return None
        except Exception as e:
            logger.error(f"✗ Could not initialize LLM client: {e}", exc_info=True)
            return None
    
    def _get_description_from_xml_db(self, procedure_id: str, title: str) -> Optional[str]:
        """
        Get description from xmlvalueprimitive database.
        
        Args:
            procedure_id: Procedure ID
            title: Procedure title for searching
            
        Returns:
            Description text or None
        """
        if not self.xml_db_connection:
            return None
        
        try:
            query_start = time.time()
            cursor = self.xml_db_connection.cursor()
            
            # Try multiple approaches
            content = None
            method_used = None
            
            # Approach 1: Try to match by procedure ID first (fastest, indexed)
            if procedure_id:
                try:
                    query = """
                        SELECT value 
                        FROM xmlvalueprimitive 
                        WHERE id = ? OR contentid = ? OR procedure_id = ?
                        LIMIT 1
                    """
                    cursor.execute(query, (procedure_id, procedure_id, procedure_id))
                    row = cursor.fetchone()
                    if row:
                        content = row[0]
                        method_used = "ID"
                        query_time = time.time() - query_start
                        if query_time > 1.0:  # Log slow queries
                            logger.debug(f"  ✓ ID search found content (length: {len(str(content))}, took {query_time:.2f}s)")
                except Exception as e:
                    logger.debug(f"  ✗ ID search failed (table may not have ID columns): {e}")
            
            # Approach 2: Use FTS if available (efficient for text search)
            if not content:
                try:
                    # Check if FTS table exists (cache this check)
                    if not hasattr(self, '_fts_table_checked'):
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = [row[0] for row in cursor.fetchall()]
                        fts_table = None
                        for table in tables:
                            if 'fts' in table.lower():
                                fts_table = table
                                break
                        self._fts_table = fts_table
                        self._fts_table_checked = True
                    else:
                        fts_table = getattr(self, '_fts_table', None)
                    
                    if fts_table:
                        # Use FTS search with title keywords
                        search_terms = title.split()[:3]  # First 3 words
                        search_query = ' OR '.join([f'"{term}"' for term in search_terms if len(term) > 2])
                        
                        query = f"""
                            SELECT value 
                            FROM {fts_table}
                            WHERE value MATCH ?
                            LIMIT 1
                        """
                        cursor.execute(query, (search_query,))
                        row = cursor.fetchone()
                        if row:
                            content = row[0]
                            method_used = "FTS"
                            query_time = time.time() - query_start
                            if query_time > 1.0:  # Log slow queries
                                logger.debug(f"  ✓ FTS search found content (length: {len(str(content))}, took {query_time:.2f}s)")
                except Exception as e:
                    logger.debug(f"  ✗ FTS search failed: {e}")
            
            if content:
                # Extract text from XML
                # Remove XML tags
                text = re.sub(r'<[^>]+>', ' ', str(content))
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                # Remove common XML artifacts
                text = re.sub(r'&[a-z]+;', ' ', text)  # Remove entities like &nbsp;
                
                if len(text) > 50:  # Only return if substantial content
                    summary = self._create_summary(text)
                    total_time = time.time() - query_start
                    if total_time > 2.0:  # Log slow extractions
                        logger.debug(f"  ✓ XML DB extraction complete (method: {method_used}, summary: {len(summary)} chars, took {total_time:.2f}s)")
                    return summary
                else:
                    logger.debug(f"  ✗ Extracted text too short ({len(text)} chars), skipping")
            else:
                query_time = time.time() - query_start
                if query_time > 2.0:  # Log slow failed queries
                    logger.debug(f"  ✗ XML DB: No content found (took {query_time:.2f}s) for procedure_id={procedure_id}, title='{title[:50]}...'")
            
            return None
        except Exception as e:
            logger.warning(f"  ✗ Error querying XML database for {procedure_id}: {e}", exc_info=True)
            return None
    
    def _get_description_from_llm(self, title: str, available_text: Optional[str] = None) -> Optional[str]:
        """
        Generate description using LLM.
        
        Args:
            title: Procedure title
            available_text: Optional available text to summarize
            
        Returns:
            Generated description or None
        """
        if not self.llm_client:
            return None
        
        try:
            llm_start = time.time()
            
            # Create messages for LLM (format expected by LLM clients)
            if available_text:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert automotive technician. Generate brief, technical descriptions of repair procedures."
                    },
                    {
                        "role": "user",
                        "content": f"""Generate a brief description (max {self.max_description_length} characters) for this automotive repair procedure:

Title: {title}

Available context:
{available_text[:1000]}

Description:"""
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert automotive technician. Generate brief, technical descriptions of repair procedures based on their titles."
                    },
                    {
                        "role": "user",
                        "content": f"""Generate a brief description (max {self.max_description_length} characters) for this automotive repair procedure based on its title:

Title: {title}

Description:"""
                    }
                ]
            
            # Call LLM
            response = self.llm_client.generate(messages, max_tokens=200, temperature=0.3)
            llm_time = time.time() - llm_start
            
            if response and len(response) > 20:
                # Truncate to max length
                original_len = len(response)
                if len(response) > self.max_description_length:
                    response = response[:self.max_description_length - 3] + "..."
                result = response.strip()
                if llm_time > 5.0:  # Log slow LLM calls
                    logger.debug(f"  ✓ LLM generated description (original: {original_len} chars, final: {len(result)} chars, took {llm_time:.2f}s)")
                # Reset failure count on success
                self.llm_failure_count = 0
                return result
            else:
                logger.debug(f"  ✗ LLM response too short or empty (length: {len(response) if response else 0}, took {llm_time:.2f}s)")
                self.llm_failure_count += 1
            
            return None
        except Exception as e:
            self.llm_failure_count += 1
            self.stats['llm_errors'] += 1
            error_msg = str(e)
            
            # Check for specific model not found errors
            if "404" in error_msg and ("not found" in error_msg.lower() or "models/" in error_msg):
                if self.llm_failure_count == 1:
                    # Only log the full error once
                    # Get the actual model name from the client
                    model_name = getattr(self.llm_client, 'model_name', 'unknown')
                    logger.error(f"  ✗ LLM Model Error: {error_msg}")
                    logger.error(f"  ✗ The model '{model_name}' is not available with API v1beta.")
                    logger.error(f"  ✗ Please update GEMINI_MODEL in your .env file to one of:")
                    logger.error(f"     - gemini-1.5-pro (recommended)")
                    logger.error(f"     - gemini-1.5-flash (faster, cheaper)")
                    logger.error(f"     - gemini-2.5-flash (if available)")
                    logger.error(f"     - gemini-1.0-pro")
                    logger.error(f"  ✗ Disabling LLM generation to prevent further errors...")
                # Disable LLM immediately for model errors
                self.llm_disabled = True
                return None
            elif self.llm_failure_count <= 3:
                # Log first few errors, then suppress
                logger.warning(f"  ✗ LLM error ({self.llm_failure_count}/{self.llm_max_failures}): {error_msg[:100]}")
                if self.llm_failure_count >= self.llm_max_failures:
                    logger.error(f"  ✗ Disabling LLM after {self.llm_failure_count} consecutive failures")
                    self.llm_disabled = True
            else:
                # Suppress further error logging
                pass
            
            return None
    
    def extract_from_ista_db(self) -> None:
        """Extract titles from ISTA database."""
        logger.info("=" * 60)
        logger.info("Extracting titles from ISTA database...")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  - Include descriptions: {self.include_descriptions}")
        if self.include_descriptions:
            logger.info(f"  - Use XML database: {self.use_xml_db}")
            logger.info(f"  - Use LLM: {self.use_llm}")
            if self.use_llm:
                logger.info(f"  - LLM provider: {self.llm_provider}")
            logger.info(f"  - Max description length: {self.max_description_length} chars")
        logger.info("=" * 60)
        
        # Initialize XML DB if needed
        if self.include_descriptions and self.use_xml_db:
            logger.info("Initializing XML database connection...")
            self.xml_db_connection = self._init_xml_db()
            if self.xml_db_connection:
                logger.info("✓ XML database ready")
            else:
                logger.warning("✗ XML database not available")
        
        # Initialize LLM if needed
        if self.include_descriptions and self.use_llm:
            logger.info("Initializing LLM client...")
            self.llm_client = self._init_llm()
            if self.llm_client:
                logger.info("✓ LLM client ready")
            else:
                logger.warning("✗ LLM client not available")
        
        try:
            ista_db = IstaDatabase()
            
            # Query all repair procedures
            with ista_db._connection.session() as session:
                from sqlalchemy import text
                
                # Get all procedures with titles
                result = session.execute(
                    text("""
                        SELECT DISTINCT 
                            ID,
                            NAME,
                            TITLE_ENGB
                        FROM XEP_INFOOBJECTS
                        WHERE TITLE_ENGB IS NOT NULL 
                           OR NAME IS NOT NULL
                    """)
                )
                
                rows = result.fetchall()
                logger.info(f"Found {len(rows)} procedures in ISTA database")
                
                # Progress tracking
                start_time = time.time()
                processed = 0
                descriptions_found = 0
                skipped_duplicates = 0
                log_interval = max(500, len(rows) // 50)  # Log every 2% or every 500, whichever is larger
                
                logger.info(f"  Starting to process {len(rows):,} procedures...")
                logger.info(f"  Progress logging every {log_interval:,} procedures (~{100/log_interval*log_interval:.1f}% intervals)")
                
                for row in rows:
                    # Check for shutdown request
                    if self._shutdown_requested:
                        logger.warning("  ⚠️  Shutdown requested, saving checkpoint and exiting...")
                        self.checkpoint()
                        break
                    
                    processed += 1
                    
                    # Periodic checkpointing
                    time_since_checkpoint = time.time() - self.last_checkpoint_time
                    if (processed % self.checkpoint_row_interval == 0 or 
                        time_since_checkpoint >= self.checkpoint_interval):
                        self.checkpoint()
                    
                    if processed % log_interval == 0:
                        elapsed = time.time() - start_time
                        pct = (processed / len(rows)) * 100
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = (len(rows) - processed) / rate if rate > 0 else 0
                        logger.info(f"  Progress: {processed:,}/{len(rows):,} ({pct:.1f}%) - "
                                  f"Descriptions: {descriptions_found:,} - "
                                  f"Skipped duplicates: {skipped_duplicates:,} - "
                                  f"Rate: {rate:.0f} rows/sec - "
                                  f"ETA: {remaining/60:.1f} min")
                    # Extract data from row
                    if hasattr(row, '_mapping'):
                        row_dict = dict(row._mapping)
                    elif hasattr(row, '_asdict'):
                        row_dict = row._asdict()
                    else:
                        # Fallback: assume tuple-like
                        row_dict = {
                            'ID': row[0] if len(row) > 0 else None,
                            'NAME': row[1] if len(row) > 1 else None,
                            'TITLE_ENGB': row[2] if len(row) > 2 else None
                        }
                    
                    procedure_id = str(row_dict.get('ID', ''))
                    name = str(row_dict.get('NAME', '')).strip()
                    title_engb = str(row_dict.get('TITLE_ENGB', '')).strip()
                    
                    # Use TITLE_ENGB if available, otherwise NAME
                    title = title_engb if title_engb else name
                    
                    if title:
                        # Normalize title
                        normalized_title = self._normalize_title(title)
                        
                        if normalized_title:
                            # Check if we already have this title with a description (incremental mode)
                            existing_metadata = self.title_to_procedure.get(normalized_title)
                            if existing_metadata and existing_metadata.get('description'):
                                # Skip expensive extraction - we already have a description
                                skipped_duplicates += 1
                                self.stats['skipped_with_description'] += 1
                                if skipped_duplicates % 1000 == 0:
                                    logger.info(f"  Skipped {skipped_duplicates:,} titles with existing descriptions (incremental mode)")
                                # Still add the title to ensure we track all procedure IDs, but reuse description
                                metadata = {
                                    'procedure_id': procedure_id,
                                    'original_title': title,
                                    'name': name,
                                    'title_engb': title_engb,
                                    'source': 'ista_db',
                                    'description': existing_metadata.get('description'),  # Reuse existing description
                                    'description_source': existing_metadata.get('description_source', 'existing')
                                }
                                self._add_title(normalized_title, metadata)
                                self.stats['ista_titles'] += 1
                                continue  # Skip expensive description extraction
                            
                            metadata = {
                                'procedure_id': procedure_id,
                                'original_title': title,
                                'name': name,
                                'title_engb': title_engb,
                                'source': 'ista_db'
                            }
                            
                            # Try to get description if needed
                            if self.include_descriptions:
                                description = None
                                description_source = None
                                
                                # Try XML database first
                                if self.use_xml_db and self.xml_db_connection:
                                    if processed % 100 == 0:  # Log every 100 attempts
                                        logger.debug(f"  Attempting XML DB extraction for procedure_id={procedure_id}, title='{title[:60]}...'")
                                    description = self._get_description_from_xml_db(procedure_id, title)
                                    if description:
                                        self.stats['from_xml_db'] += 1
                                        description_source = "XML DB"
                                        if self.stats['from_xml_db'] % 100 == 0:
                                            logger.info(f"  ✓ XML DB: Found {self.stats['from_xml_db']:,} descriptions so far")
                                    else:
                                        if processed % 1000 == 0:  # Log less frequently for failures
                                            logger.debug(f"  ✗ XML DB: No description for procedure_id={procedure_id}")
                                
                                # Try LLM if still no description and LLM is not disabled
                                if not description and self.use_llm and self.llm_client and not self.llm_disabled:
                                    if processed % 100 == 0:  # Log every 100 attempts
                                        logger.debug(f"  Attempting LLM generation for procedure_id={procedure_id}, title='{title[:60]}...'")
                                    description = self._get_description_from_llm(title)
                                    if description:
                                        self.stats['from_llm'] += 1
                                        description_source = "LLM"
                                        self.llm_failure_count = 0  # Reset on success
                                        if self.stats['from_llm'] % 10 == 0:  # Log every 10 LLM successes
                                            logger.info(f"  ✓ LLM: Generated {self.stats['from_llm']:,} descriptions so far")
                                    else:
                                        if processed % 1000 == 0:  # Log less frequently for failures
                                            logger.debug(f"  ✗ LLM: No description for procedure_id={procedure_id}")
                                
                                if description:
                                    metadata['description'] = description
                                    metadata['description_source'] = description_source
                                    descriptions_found += 1
                                    if descriptions_found % 1000 == 0:
                                        logger.info(f"  ✓ Total descriptions found: {descriptions_found:,} ({description_source})")
                                else:
                                    if processed % 5000 == 0:  # Log every 5000 failures
                                        logger.debug(f"  ✗ No description found for procedure_id={procedure_id}, title='{title[:60]}...'")
                            
                            # Check if this is a new title (not in existing data)
                            is_new = normalized_title not in self.titles
                            if is_new:
                                self.stats['new_titles'] += 1
                            
                            self._add_title(normalized_title, metadata)
                            self.stats['ista_titles'] += 1
                            
                            # Track if we got a new description
                            if description and is_new:
                                self.stats['new_descriptions'] += 1
                
                total_time = time.time() - start_time
                logger.info(f"✓ Extracted {self.stats['ista_titles']:,} titles from ISTA database (took {total_time/60:.1f} minutes)")
                logger.info(f"  Processing rate: {len(rows)/total_time:.0f} rows/sec")
                if self.include_descriptions:
                    logger.info(f"  Descriptions found: {descriptions_found:,} total")
                    logger.info(f"    - {self.stats['from_xml_db']:,} from XML database")
                    logger.info(f"    - {self.stats['from_llm']:,} from LLM")
                    logger.info(f"    - {skipped_duplicates:,} skipped (already had descriptions)")
                    if descriptions_found < self.stats['ista_titles']:
                        missing = self.stats['ista_titles'] - descriptions_found
                        pct = (descriptions_found / self.stats['ista_titles']) * 100
                        logger.warning(f"    - {missing:,} titles still missing descriptions ({pct:.1f}% coverage)")
                
        except Exception as e:
            logger.error(f"Error extracting from ISTA database: {e}", exc_info=True)
        finally:
            # Close XML DB connection
            if self.xml_db_connection:
                self.xml_db_connection.close()
                self.xml_db_connection = None
    
    def extract_from_vector_store(self, retrieval_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Extract titles from vector store.
        
        Args:
            retrieval_config: Optional retrieval configuration dict
        """
        logger.info(f"Extracting titles from vector store (include_descriptions={self.include_descriptions})...")
        
        try:
            # Load retrieval config if not provided
            if retrieval_config is None:
                paths = get_paths()
                with open(paths.retrieval_config, 'r', encoding='utf-8') as f:
                    retrieval_config = yaml.safe_load(f)
            
            # Initialize vector store
            vector_store = VectorStore(retrieval_config.get("vector_store", {}))
            
            # Get all points from collection via scroll
            try:
                offset = None
                batch_size = 1000
                total_scrolled = 0

                while True:
                    scroll_result = vector_store.scroll(
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    
                    points = scroll_result[0]  # Points
                    next_offset = scroll_result[1]  # Next offset
                    
                    if not points:
                        break
                    
                    # Extract titles from points
                    for point in points:
                        payload = point.payload or {}
                        
                        # Get title or procedure_name
                        title = payload.get('title') or payload.get('procedure_name', '')
                        procedure_id = payload.get('procedure_id') or str(point.id)
                        
                        if title:
                            normalized_title = self._normalize_title(title)
                            
                            if normalized_title:
                                # Get description/summary from text if available
                                description = None
                                text_content = payload.get('text', '')
                                if text_content:
                                    # Create a summary (first 500 chars or first paragraph)
                                    description = self._create_summary(text_content)
                                
                                metadata = {
                                    'procedure_id': procedure_id,
                                    'original_title': title,
                                    'source': 'vector_store',
                                    'payload': {k: v for k, v in payload.items() if k not in ['text']}  # Exclude large text
                                }
                                
                                # Add description if available and include_descriptions is True
                                if description and hasattr(self, 'include_descriptions') and self.include_descriptions:
                                    metadata['description'] = description
                                    metadata['description_length'] = len(text_content)
                                
                                self._add_title(normalized_title, metadata)
                                self.stats['vector_store_titles'] += 1
                    
                    total_scrolled += len(points)
                    
                    # Check if we're done
                    if next_offset is None:
                        break
                    offset = next_offset
                
                logger.info(f"Extracted {self.stats['vector_store_titles']} titles from vector store (scrolled {total_scrolled} points)")
                
            except Exception as e:
                logger.warning(f"Could not scroll vector store (may be empty or not accessible): {e}")
                logger.info("Skipping vector store extraction")
                
        except Exception as e:
            logger.error(f"Error extracting from vector store: {e}", exc_info=True)
    
    def _normalize_title(self, title: str) -> Optional[str]:
        """
        Normalize title for matching.
        
        Args:
            title: Raw title string
            
        Returns:
            Normalized title or None if invalid
        """
        if not title or not isinstance(title, str):
            return None
        
        # Strip whitespace
        title = title.strip()
        
        # Remove if too short
        if len(title) < 3:
            return None
        
        # Remove if too long (likely not a title)
        if len(title) > 200:
            return None
        
        return title
    
    def _create_summary(self, text: str) -> str:
        """
        Create a summary/description from full text.
        
        Args:
            text: Full text content
            
        Returns:
            Summary string (truncated to max_description_length)
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Try to get first sentence or paragraph
        # Split by sentence endings
        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            # Take first sentence(s) up to max length
            summary = sentences[0]
            for sentence in sentences[1:]:
                if len(summary) + len(sentence) + 1 <= self.max_description_length:
                    summary += ". " + sentence
                else:
                    break
            
            # Truncate if still too long
            if len(summary) > self.max_description_length:
                summary = summary[:self.max_description_length - 3] + "..."
            
            return summary
        
        # Fallback: just truncate
        if len(text) > self.max_description_length:
            return text[:self.max_description_length - 3] + "..."
        
        return text
    
    def _load_existing_csv(self, csv_path: Path) -> None:
        """
        Load existing CSV file for incremental processing.
        
        Args:
            csv_path: Path to existing CSV file
        """
        logger.info(f"Loading existing data from: {csv_path}")
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                loaded_count = 0
                with_description_count = 0
                
                for row in reader:
                    title = row.get('title', '').strip()
                    if not title:
                        continue
                    
                    # Normalize title
                    normalized_title = self._normalize_title(title)
                    if not normalized_title:
                        continue
                    
                    # Build metadata
                    metadata = {
                        'procedure_id': row.get('procedure_id', ''),
                        'source': row.get('source', 'existing'),
                        'original_title': title
                    }
                    
                    # Add description if present
                    description = row.get('description', '').strip()
                    if description:
                        metadata['description'] = description
                        with_description_count += 1
                    
                    # Add to collections
                    self.titles.add(normalized_title)
                    self.title_to_procedure[normalized_title] = metadata
                    loaded_count += 1
                    
                    if description:
                        self.stats['with_descriptions'] += 1
                
                self.stats['loaded_from_existing'] = loaded_count
                logger.info(f"  ✓ Loaded {loaded_count:,} titles from existing file")
                logger.info(f"    - {with_description_count:,} titles already have descriptions")
                logger.info(f"    - {loaded_count - with_description_count:,} titles need descriptions")
        except Exception as e:
            logger.warning(f"  ✗ Could not load existing CSV file: {e}")
            logger.warning(f"    Starting fresh (not incremental)")
    
    def _add_title(self, title: str, metadata: Dict[str, Any]) -> None:
        """
        Add title to collection, handling duplicates.
        
        Args:
            title: Normalized title
            metadata: Title metadata
        """
        if title in self.titles:
            self.stats['duplicates'] += 1
            # Update metadata if we have more info
            existing = self.title_to_procedure.get(title, {})
            if metadata.get('source') == 'ista_db' and existing.get('source') != 'ista_db':
                # Prefer ISTA DB source
                self.title_to_procedure[title] = metadata
            elif metadata.get('description') and not existing.get('description'):
                # Add description if we have one and existing doesn't
                existing.update(metadata)
                self.title_to_procedure[title] = existing
        else:
            self.titles.add(title)
            self.title_to_procedure[title] = metadata
            if metadata.get('description'):
                self.stats['with_descriptions'] += 1
    
    def get_titles(self) -> List[str]:
        """
        Get sorted list of all unique titles.
        
        Returns:
            Sorted list of titles
        """
        return sorted(list(self.titles))
    
    def get_titles_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Get titles with metadata.
        
        Returns:
            List of dicts with title and metadata
        """
        return [
            {
                'title': title,
                **metadata
            }
            for title, metadata in sorted(self.title_to_procedure.items())
        ]
    
    def export_to_json(self, output_path: Path, include_metadata: bool = False) -> None:
        """
        Export titles to JSON file.
        
        Args:
            output_path: Path to output file
            include_metadata: If True, include metadata for each title
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if include_metadata:
            data = {
                'titles': self.get_titles_with_metadata(),
                'statistics': self.stats,
                'total_unique_titles': len(self.titles)
            }
        else:
            data = {
                'titles': self.get_titles(),
                'statistics': self.stats,
                'total_unique_titles': len(self.titles)
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(self.titles)} titles to {output_path}")
    
    def export_to_text(self, output_path: Path) -> None:
        """
        Export titles to simple text file (one per line).
        
        Args:
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for title in self.get_titles():
                f.write(f"{title}\n")
        
        logger.info(f"Exported {len(self.titles)} titles to {output_path}")
    
    def checkpoint(self, output_path: Optional[Path] = None) -> None:
        """
        Save current progress to checkpoint file.
        
        Args:
            output_path: Path to save checkpoint (uses self.checkpoint_path if None)
        """
        if not self.checkpoint_enabled:
            return
        
        checkpoint_file = output_path or self.checkpoint_path
        if not checkpoint_file:
            return
        
        try:
            logger.info(f"  💾 Saving checkpoint to {checkpoint_file}...")
            self.export_to_csv(checkpoint_file)
            self.last_checkpoint_time = time.time()
            logger.info(f"  ✓ Checkpoint saved ({len(self.titles):,} titles)")
        except Exception as e:
            logger.error(f"  ✗ Failed to save checkpoint: {e}", exc_info=True)
    
    def export_to_csv(self, output_path: Path, append: bool = False) -> None:
        """
        Export titles to CSV file.
        
        Args:
            output_path: Path to output CSV file
            append: If True, append to existing file (for incremental writes)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        titles_with_metadata = self.get_titles_with_metadata()
        
        # Determine CSV columns
        if self.include_descriptions:
            fieldnames = ['title', 'procedure_id', 'description', 'source']
        else:
            fieldnames = ['title', 'procedure_id', 'source']
        
        # Use append mode if requested and file exists
        file_mode = 'a' if append and output_path.exists() else 'w'
        write_header = not (append and output_path.exists())
        
        with open(output_path, file_mode, encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            
            for item in titles_with_metadata:
                row = {
                    'title': item.get('title', ''),
                    'procedure_id': item.get('procedure_id', ''),
                    'source': item.get('source', '')
                }
                
                if self.include_descriptions:
                    row['description'] = item.get('description', '')
                
                writer.writerow(row)
        
        if not append:
            logger.info(f"Exported {len(titles_with_metadata)} titles to CSV: {output_path}")
            if self.include_descriptions:
                logger.info(f"  - {self.stats['with_descriptions']} titles include descriptions")
    
    def export_for_scraping_agent(self, output_path: Path) -> None:
        """
        Export titles in format optimized for scraping agent matching.
        
        Includes:
        - Simple title list (for exact/fuzzy matching)
        - Title variations (normalized versions)
        - Common keywords extracted from titles
        - Descriptions (if include_descriptions=True)
        - Statistics
        
        Args:
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        titles = self.get_titles()
        titles_with_metadata = self.get_titles_with_metadata()
        
        # Extract common keywords from titles
        keywords = self._extract_keywords(titles)
        
        # Group titles by category (if possible)
        categories = self._categorize_titles(titles)
        
        # Build titles data structure
        if self.include_descriptions:
            # Include descriptions in titles data
            titles_data = [
                {
                    'title': item['title'],
                    'procedure_id': item.get('procedure_id', ''),
                    'description': item.get('description', ''),
                    'source': item.get('source', '')
                }
                for item in titles_with_metadata
            ]
        else:
            # Just titles
            titles_data = titles
        
        data = {
            'format_version': '1.0',
            'description': 'Valid repair guide titles for MIST system - use for fuzzy matching scraped content',
            'includes_descriptions': self.include_descriptions,
            'statistics': {
                **self.stats,
                'total_unique_titles': len(titles),
                'total_keywords': len(keywords),
                'categories': {cat: len(titles) for cat, titles in categories.items()}
            },
            'titles': titles_data,
            'keywords': sorted(keywords),
            'categories': categories,
            'matching_instructions': {
                'exact_match': 'Try exact string matching first',
                'fuzzy_match': 'Use fuzzy string matching (e.g., Levenshtein distance, ratio > 0.8)',
                'keyword_match': 'Match if scraped content contains 2+ keywords from a title',
                'description_match': 'If descriptions are included, match against description text as well',
                'normalization': 'Normalize both scraped content and titles before matching (lowercase, remove punctuation)'
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(titles)} titles for scraping agent to {output_path}")
        if self.include_descriptions:
            logger.info(f"  - {self.stats['with_descriptions']} titles include descriptions")
    
    def _extract_keywords(self, titles: List[str]) -> Set[str]:
        """
        Extract common keywords from titles.
        
        Args:
            titles: List of titles
            
        Returns:
            Set of keywords
        """
        keywords = set()
        
        # Common automotive keywords to look for
        automotive_terms = [
            'diagnosis', 'repair', 'replace', 'check', 'inspect', 'test',
            'misfire', 'sensor', 'coil', 'plug', 'injector', 'valve',
            'gasket', 'seal', 'filter', 'pump', 'motor', 'actuator',
            'circuit', 'wiring', 'connector', 'fuse', 'relay', 'module',
            'calibration', 'reset', 'programming', 'coding', 'update'
        ]
        
        # Extract words from titles
        import re
        for title in titles:
            # Split into words
            words = re.findall(r'\b\w+\b', title.lower())
            
            # Add significant words (length > 3, not common stop words)
            stop_words = {'the', 'and', 'or', 'for', 'with', 'from', 'this', 'that', 'are', 'was', 'were'}
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    keywords.add(word)
        
        # Add known automotive terms
        keywords.update(automotive_terms)
        
        return keywords
    
    def _categorize_titles(self, titles: List[str]) -> Dict[str, List[str]]:
        """
        Categorize titles by common patterns.
        
        Args:
            titles: List of titles
            
        Returns:
            Dict mapping category to list of titles
        """
        categories = defaultdict(list)
        
        # Category patterns
        patterns = {
            'misfire': ['misfire', 'cylinder'],
            'sensor': ['sensor', 'maf', 'o2', 'oxygen'],
            'ignition': ['ignition', 'coil', 'spark', 'plug'],
            'fuel': ['fuel', 'injector', 'pump', 'filter'],
            'emissions': ['emission', 'catalyst', 'egr'],
            'transmission': ['transmission', 'gearbox', 'clutch'],
            'electrical': ['electrical', 'wiring', 'circuit', 'fuse', 'relay'],
            'diagnosis': ['diagnosis', 'diagnostic', 'test', 'check'],
            'repair': ['repair', 'replace', 'install', 'remove'],
            'calibration': ['calibration', 'reset', 'programming', 'coding']
        }
        
        for title in titles:
            title_lower = title.lower()
            categorized = False
            
            for category, keywords in patterns.items():
                if any(keyword in title_lower for keyword in keywords):
                    categories[category].append(title)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(title)
        
        return dict(categories)
    
    def print_statistics(self) -> None:
        """Print extraction statistics."""
        logger.info("=" * 60)
        logger.info("Extraction Statistics")
        logger.info("=" * 60)
        logger.info(f"ISTA DB titles: {self.stats['ista_titles']}")
        logger.info(f"Vector store titles: {self.stats['vector_store_titles']}")
        logger.info(f"Duplicates removed: {self.stats['duplicates']}")
        if self.stats.get('loaded_from_existing', 0) > 0:
            logger.info(f"Loaded from existing file: {self.stats['loaded_from_existing']}")
        if self.stats.get('skipped_with_description', 0) > 0:
            logger.info(f"Skipped (already had descriptions): {self.stats['skipped_with_description']}")
        if self.stats.get('new_titles', 0) > 0:
            logger.info(f"New titles added: {self.stats['new_titles']}")
        logger.info(f"Total unique titles: {len(self.titles)}")
        if self.include_descriptions:
            logger.info(f"Titles with descriptions: {self.stats['with_descriptions']}")
            if self.stats.get('from_xml_db', 0) > 0:
                logger.info(f"  - From XML database: {self.stats['from_xml_db']}")
            if self.stats.get('from_llm', 0) > 0:
                logger.info(f"  - From LLM: {self.stats['from_llm']}")
            if self.stats.get('new_descriptions', 0) > 0:
                logger.info(f"  - New descriptions added: {self.stats['new_descriptions']}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract valid repair guide titles from MIST database and vector store"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file path (default: data/training/valid_repair_guide_titles.json)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'text', 'agent', 'csv'],
        default='agent',
        help='Output format: json (with metadata), text (simple list), agent (optimized for scraping agent), csv (CSV format)'
    )
    parser.add_argument(
        '--include-metadata',
        action='store_true',
        help='Include metadata in JSON output (only applies to json format)'
    )
    parser.add_argument(
        '--skip-vector-store',
        action='store_true',
        help='Skip vector store extraction (only extract from ISTA DB)'
    )
    parser.add_argument(
        '--include-descriptions',
        action='store_true',
        help='Include procedure descriptions/summaries (from vector store text content)'
    )
    parser.add_argument(
        '--max-description-length',
        type=int,
        default=500,
        help='Maximum length of description summary in characters (default: 500)'
    )
    parser.add_argument(
        '--use-xml-db',
        action='store_true',
        default=True,
        help='Try to extract descriptions from xmlvalueprimitive database (default: True)'
    )
    parser.add_argument(
        '--no-xml-db',
        action='store_false',
        dest='use_xml_db',
        help='Disable XML database extraction'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM to generate descriptions for missing entries (requires API keys)'
    )
    parser.add_argument(
        '--llm-provider',
        choices=['openai', 'anthropic', 'open_source', 'gemini'],
        default='gemini',
        help='LLM provider to use for generating descriptions (default: gemini)'
    )
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Incremental mode: load existing output file and only process titles without descriptions (saves LLM calls)'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        paths = get_paths()
        output_dir = paths.mist_root / "data" / "training"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'agent':
            args.output = output_dir / "valid_repair_guide_titles.json"
        elif args.format == 'json':
            args.output = output_dir / "repair_guide_titles_with_metadata.json"
        elif args.format == 'csv':
            args.output = output_dir / "valid_repair_guide_titles.csv"
        else:
            args.output = output_dir / "repair_guide_titles.txt"
    
    # Determine incremental file path
    incremental_file = None
    if args.incremental:
        # Use the output file as the incremental source
        incremental_file = args.output
        if incremental_file and incremental_file.exists():
            logger.info(f"Incremental mode: Will load existing data from {incremental_file}")
            logger.info(f"  - Will skip titles that already have descriptions")
            logger.info(f"  - Will only process new titles or titles without descriptions")
        else:
            logger.info(f"Incremental mode: Output file doesn't exist yet, starting fresh")
            incremental_file = None  # Don't try to load non-existent file
    
    # Extract titles
    extractor = RepairGuideTitleExtractor(
        include_descriptions=args.include_descriptions,
        max_description_length=args.max_description_length,
        use_xml_db=args.use_xml_db,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider if args.use_llm else None,
        incremental_file=incremental_file
    )
    
    # Set up checkpoint path (use output file for CSV format, otherwise use a checkpoint file)
    if args.format == 'csv':
        extractor.checkpoint_path = args.output
    else:
        # For other formats, create a checkpoint CSV file
        extractor.checkpoint_path = args.output.with_suffix('.checkpoint.csv')
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.warning(f"\n⚠️  Received signal {signum}, saving checkpoint and exiting...")
        extractor._shutdown_requested = True
        extractor.checkpoint()
        logger.info("✓ Checkpoint saved. Exiting gracefully.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Set up atexit handler to save on normal exit
    def save_on_exit():
        if extractor.checkpoint_path and extractor.checkpoint_enabled:
            try:
                extractor.checkpoint()
            except Exception as e:
                logger.error(f"Error saving checkpoint on exit: {e}")
    
    atexit.register(save_on_exit)
    
    try:
        # Extract from ISTA DB
        extractor.extract_from_ista_db()
        
        # Extract from vector store (if not skipped)
        if not args.skip_vector_store:
            extractor.extract_from_vector_store()
        
        # Final checkpoint before export
        if extractor.checkpoint_path:
            extractor.checkpoint()
        
        # Print statistics
        extractor.print_statistics()
        
        # Export based on format
        if args.format == 'agent':
            extractor.export_for_scraping_agent(args.output)
        elif args.format == 'json':
            extractor.export_to_json(args.output, include_metadata=args.include_metadata)
        elif args.format == 'csv':
            # For CSV, the checkpoint is the final file, so we're done
            logger.info(f"✓ Final export complete. Output saved to: {args.output}")
        else:
            extractor.export_to_text(args.output)
            logger.info(f"Extraction complete. Output saved to: {args.output}")
        
        # Remove checkpoint file if it exists and we're using CSV (since final file is the checkpoint)
        if args.format == 'csv' and extractor.checkpoint_path and extractor.checkpoint_path.exists():
            # Checkpoint is the same as output, so no cleanup needed
            pass
        elif extractor.checkpoint_path and extractor.checkpoint_path.exists():
            # Remove checkpoint file on successful completion
            try:
                extractor.checkpoint_path.unlink()
                logger.info(f"✓ Removed checkpoint file: {extractor.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint file: {e}")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user, saving checkpoint...")
        extractor.checkpoint()
        logger.info("✓ Checkpoint saved. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        logger.warning("Attempting to save checkpoint before exiting...")
        extractor.checkpoint()
        raise


if __name__ == '__main__':
    main()
