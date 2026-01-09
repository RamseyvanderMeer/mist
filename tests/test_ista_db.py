"""
Unit tests for IstaDatabase class.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker

from src.database.ista_db import IstaDatabase
from src.database.connection import DatabaseConnection


class TestIstaDatabase:
    """Test IstaDatabase methods."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary SQLite database."""
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
            db_path = f.name
        
        # Create tables for testing
        engine = create_engine(f'sqlite:///{db_path}')
        with engine.connect() as conn:
            # Create minimal schema for testing
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS XEP_FAULTCODES (
                    ID TEXT PRIMARY KEY,
                    CODE TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS RG_ECUFAULT_DOCIDS (
                    FAULTCODE_ID TEXT,
                    DOCID TEXT,
                    FOREIGN KEY (FAULTCODE_ID) REFERENCES XEP_FAULTCODES(ID)
                )
            """))
            conn.commit()
        
        yield db_path
        Path(db_path).unlink(missing_ok=True)
    
    def test_get_fault_codes_for_procedure(self, temp_db):
        """Test get_fault_codes_for_procedure method."""
        # Insert test data
        engine = create_engine(f'sqlite:///{temp_db}')
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO XEP_FAULTCODES (ID, CODE) VALUES
                ('fault1', 'P0301'),
                ('fault2', 'P0302')
            """))
            conn.execute(text("""
                INSERT INTO RG_ECUFAULT_DOCIDS (FAULTCODE_ID, DOCID) VALUES
                ('fault1', 'proc1'),
                ('fault2', 'proc1')
            """))
            conn.commit()
        
        # Test the method
        with patch('src.database.ista_db.create_connection') as mock_create:
            mock_connection = Mock()
            mock_session = MagicMock()
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=False)
            
            # Mock the query result
            mock_row1 = Mock()
            mock_row1.__getitem__ = Mock(return_value='P0301')
            mock_row1.CODE = 'P0301'
            
            mock_row2 = Mock()
            mock_row2.__getitem__ = Mock(return_value='P0302')
            mock_row2.CODE = 'P0302'
            
            mock_result = Mock()
            mock_result.fetchall = Mock(return_value=[
                ('P0301',), ('P0302',)
            ])
            mock_session.execute = Mock(return_value=mock_result)
            
            mock_connection.session = Mock(return_value=mock_session)
            mock_create.return_value = mock_connection
            
            ista_db = IstaDatabase(db_path=temp_db)
            fault_codes = ista_db.get_fault_codes_for_procedure('proc1')
            
            assert len(fault_codes) == 2
            assert 'P0301' in fault_codes
            assert 'P0302' in fault_codes
    
    def test_get_fault_codes_for_procedure_no_results(self, temp_db):
        """Test get_fault_codes_for_procedure with no matching fault codes."""
        with patch('src.database.ista_db.create_connection') as mock_create:
            mock_connection = Mock()
            mock_session = MagicMock()
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=False)
            
            mock_result = Mock()
            mock_result.fetchall = Mock(return_value=[])
            mock_session.execute = Mock(return_value=mock_result)
            
            mock_connection.session = Mock(return_value=mock_session)
            mock_create.return_value = mock_connection
            
            ista_db = IstaDatabase(db_path=temp_db)
            fault_codes = ista_db.get_fault_codes_for_procedure('proc999')
            
            assert len(fault_codes) == 0
            assert fault_codes == []
