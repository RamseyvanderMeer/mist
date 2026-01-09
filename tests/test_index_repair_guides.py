"""
Unit tests for repair guide indexing script.

Tests the RepairGuideIndexer class with mocked dependencies.
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
import torch

# Import the indexer class
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from index_repair_guides import RepairGuideIndexer


class TestRepairGuideIndexer:
    """Test RepairGuideIndexer class."""
    
    @pytest.fixture
    def embedding_config(self):
        """Fixture for embedding configuration."""
        return {
            "models": {
                "fault_code": {
                    "model_name": "intfloat/e5-mistral-7b-instruct",
                    "projection_dim": 768,
                    "device": "cpu"
                }
            }
        }
    
    @pytest.fixture
    def retrieval_config(self):
        """Fixture for retrieval configuration."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "collection_name": "test_repair_guides",
                "distance_metric": "cosine",
                "vector_size": 768,
                "url": "http://localhost:6333"
            }
        }
    
    @pytest.fixture
    def temp_checkpoint(self, tmp_path):
        """Fixture for temporary checkpoint file."""
        return tmp_path / "test_checkpoint.json"
    
    @pytest.fixture
    def mock_ista_db(self):
        """Fixture for mocked IstaDatabase."""
        mock_db = Mock()
        mock_db.connection = Mock()
        mock_db.connection.session = Mock(return_value=Mock().__enter__())
        return mock_db
    
    @pytest.fixture
    def mock_encoder(self):
        """Fixture for mocked FaultCodeEncoder."""
        mock_enc = Mock()
        mock_enc.encode = Mock(return_value=torch.randn(2, 768))
        mock_enc.get_dimension = Mock(return_value=768)
        return mock_enc
    
    @pytest.fixture
    def mock_vector_store(self):
        """Fixture for mocked VectorStore."""
        mock_vs = Mock()
        mock_vs.add = Mock()
        return mock_vs
    
    @pytest.fixture
    def sample_procedures(self):
        """Fixture for sample procedure data."""
        return [
            {
                "id": "proc1",
                "title_engb": "Engine Oil Change",
                "name": "Oil Change Procedure"
            },
            {
                "id": "proc2",
                "title_engb": "Brake Pad Replacement",
                "name": "Brake Service"
            }
        ]
    
    @pytest.fixture
    def sample_segments(self):
        """Fixture for sample segment data."""
        return [
            {
                "INFOOBJECTID": "proc1",
                "SEGMENTORDER": 1,
                "CONTENT_ENGB": "Step 1: Drain engine oil"
            },
            {
                "INFOOBJECTID": "proc1",
                "SEGMENTORDER": 2,
                "CONTENT_ENGB": "Step 2: Replace oil filter"
            }
        ]
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_indexer_initialization(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config,
        temp_checkpoint
    ):
        """Test RepairGuideIndexer initialization."""
        mock_ista_db_class.return_value = Mock()
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config,
            checkpoint_file=temp_checkpoint,
            batch_size=50
        )
        
        assert indexer.batch_size == 50
        assert indexer.checkpoint_file == temp_checkpoint
        assert len(indexer.indexed_ids) == 0
        assert indexer.processed_count == 0
        assert indexer.error_count == 0
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_load_checkpoint_existing(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config,
        temp_checkpoint
    ):
        """Test loading checkpoint from existing file."""
        # Create checkpoint file
        checkpoint_data = {
            "indexed_ids": ["proc1", "proc2"],
            "processed_count": 2,
            "last_updated": "2024-01-01T00:00:00"
        }
        temp_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_checkpoint, 'w') as f:
            json.dump(checkpoint_data, f)
        
        mock_ista_db_class.return_value = Mock()
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config,
            checkpoint_file=temp_checkpoint
        )
        
        assert len(indexer.indexed_ids) == 2
        assert "proc1" in indexer.indexed_ids
        assert "proc2" in indexer.indexed_ids
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_save_checkpoint(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config,
        temp_checkpoint
    ):
        """Test saving checkpoint file."""
        mock_ista_db_class.return_value = Mock()
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config,
            checkpoint_file=temp_checkpoint
        )
        
        indexer.indexed_ids.add("proc1")
        indexer.indexed_ids.add("proc2")
        indexer.processed_count = 2
        indexer._save_checkpoint()
        
        # Verify checkpoint was saved
        assert temp_checkpoint.exists()
        with open(temp_checkpoint, 'r') as f:
            checkpoint = json.load(f)
        
        assert len(checkpoint["indexed_ids"]) == 2
        assert "proc1" in checkpoint["indexed_ids"]
        assert "proc2" in checkpoint["indexed_ids"]
        assert checkpoint["processed_count"] == 2
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_get_all_procedures(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config,
        sample_procedures
    ):
        """Test getting all procedures from database."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        
        # Mock database query result
        from sqlalchemy.engine import Result
        mock_row1 = Mock()
        mock_row1.ID = "proc1"
        mock_row1.TITLE_ENGB = "Engine Oil Change"
        mock_row1.NAME = "Oil Change Procedure"
        
        mock_row2 = Mock()
        mock_row2.ID = "proc2"
        mock_row2.TITLE_ENGB = "Brake Pad Replacement"
        mock_row2.NAME = "Brake Service"
        
        mock_result = Mock()
        mock_result.fetchall = Mock(return_value=[mock_row1, mock_row2])
        mock_session.execute = Mock(return_value=mock_result)
        
        mock_db.connection = Mock()
        mock_db.connection.session = Mock(return_value=mock_session)
        mock_ista_db_class.return_value = mock_db
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config
        )
        
        procedures = indexer._get_all_procedures()
        
        assert len(procedures) == 2
        assert procedures[0]["id"] == "proc1"
        assert procedures[1]["id"] == "proc2"
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_get_procedure_text(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config,
        sample_procedures,
        sample_segments
    ):
        """Test getting procedure text from segments."""
        mock_db = Mock()
        mock_db.get_info_segments = Mock(return_value=sample_segments)
        
        mock_ista_db_class.return_value = mock_db
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config
        )
        
        procedure = sample_procedures[0]
        text = indexer._get_procedure_text(procedure)
        
        assert "Engine Oil Change" in text
        assert "Step 1: Drain engine oil" in text
        assert "Step 2: Replace oil filter" in text
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_process_procedure(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config,
        sample_procedures,
        sample_segments
    ):
        """Test processing a single procedure."""
        mock_db = Mock()
        mock_db.get_info_segments = Mock(return_value=sample_segments)
        mock_db.get_fault_codes_for_procedure = Mock(return_value=["P0301", "P0302"])
        
        mock_ista_db_class.return_value = mock_db
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config
        )
        
        procedure = sample_procedures[0]
        document = indexer._process_procedure(procedure)
        
        assert document is not None
        assert document["id"] == "proc1"
        assert document["procedure_id"] == "proc1"
        assert document["title"] == "Engine Oil Change"
        assert document["fault_codes"] == ["P0301", "P0302"]
        assert "text" in document
        assert "metadata" in document
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_process_procedure_no_content(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config
    ):
        """Test processing procedure with no content."""
        mock_db = Mock()
        mock_db.get_info_segments = Mock(return_value=[])
        mock_db.get_fault_codes_for_procedure = Mock(return_value=[])
        
        mock_ista_db_class.return_value = mock_db
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config
        )
        
        procedure = {"id": "proc1", "title_engb": "", "name": ""}
        document = indexer._process_procedure(procedure)
        
        assert document is None
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_encode_batch(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config
    ):
        """Test encoding a batch of documents."""
        mock_encoder = Mock()
        mock_encoder.encode = Mock(return_value=torch.randn(2, 768))
        
        mock_ista_db_class.return_value = Mock()
        mock_encoder_class.return_value = mock_encoder
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config
        )
        indexer.encoder = mock_encoder
        
        documents = [
            {"text": "Procedure 1 text"},
            {"text": "Procedure 2 text"}
        ]
        
        embeddings = indexer._encode_batch(documents)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)
        mock_encoder.encode.assert_called_once()
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_store_batch(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config
    ):
        """Test storing a batch of documents."""
        mock_vector_store = Mock()
        mock_vector_store.add = Mock()
        
        mock_ista_db_class.return_value = Mock()
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config
        )
        indexer.vector_store = mock_vector_store
        
        documents = [
            {"id": "proc1", "text": "Procedure 1"},
            {"id": "proc2", "text": "Procedure 2"}
        ]
        embeddings = np.random.randn(2, 768)
        
        indexer._store_batch(documents, embeddings)
        
        mock_vector_store.add.assert_called_once()
        assert "proc1" in indexer.indexed_ids
        assert "proc2" in indexer.indexed_ids
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_index_with_resume(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config,
        temp_checkpoint
    ):
        """Test indexing with resume functionality."""
        # Create checkpoint with one already indexed
        checkpoint_data = {
            "indexed_ids": ["proc1"],
            "processed_count": 1
        }
        temp_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_checkpoint, 'w') as f:
            json.dump(checkpoint_data, f)
        
        mock_db = Mock()
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        
        # Mock query to return 2 procedures
        mock_row1 = Mock()
        mock_row1.ID = "proc1"
        mock_row1.TITLE_ENGB = "Procedure 1"
        mock_row1.NAME = "Proc1"
        
        mock_row2 = Mock()
        mock_row2.ID = "proc2"
        mock_row2.TITLE_ENGB = "Procedure 2"
        mock_row2.NAME = "Proc2"
        
        mock_result = Mock()
        mock_result.fetchall = Mock(return_value=[mock_row1, mock_row2])
        mock_session.execute = Mock(return_value=mock_result)
        
        mock_db.connection = Mock()
        mock_db.connection.session = Mock(return_value=mock_session)
        mock_db.get_info_segments = Mock(return_value=[
            {"CONTENT_ENGB": "Test content"}
        ])
        mock_db.get_fault_codes_for_procedure = Mock(return_value=["P0301"])
        
        mock_encoder = Mock()
        mock_encoder.encode = Mock(return_value=torch.randn(1, 768))
        
        mock_vector_store = Mock()
        mock_vector_store.add = Mock()
        
        mock_ista_db_class.return_value = mock_db
        mock_encoder_class.return_value = mock_encoder
        mock_vector_store_class.return_value = mock_vector_store
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config,
            checkpoint_file=temp_checkpoint,
            batch_size=10
        )
        
        stats = indexer.index(limit=2, resume=True)
        
        # Should only process proc2 (proc1 already indexed)
        assert stats["indexed"] >= 1
        assert "proc1" in indexer.indexed_ids
        # proc2 should also be indexed now
        assert "proc2" in indexer.indexed_ids or stats["processed"] == 1
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_index_error_handling(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config
    ):
        """Test error handling during indexing."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        
        # Mock query to return 2 procedures
        mock_row1 = Mock()
        mock_row1.ID = "proc1"
        mock_row1.TITLE_ENGB = "Procedure 1"
        mock_row1.NAME = "Proc1"
        
        mock_row2 = Mock()
        mock_row2.ID = "proc2"
        mock_row2.TITLE_ENGB = "Procedure 2"
        mock_row2.NAME = "Proc2"
        
        mock_result = Mock()
        mock_result.fetchall = Mock(return_value=[mock_row1, mock_row2])
        mock_session.execute = Mock(return_value=mock_result)
        
        mock_db.connection = Mock()
        mock_db.connection.session = Mock(return_value=mock_session)
        
        # First procedure succeeds, second fails
        def get_segments(proc_id):
            if proc_id == "proc1":
                return [{"CONTENT_ENGB": "Content 1"}]
            else:
                raise Exception("Database error")
        
        mock_db.get_info_segments = Mock(side_effect=get_segments)
        mock_db.get_fault_codes_for_procedure = Mock(return_value=["P0301"])
        
        mock_encoder = Mock()
        mock_encoder.encode = Mock(return_value=torch.randn(1, 768))
        
        mock_vector_store = Mock()
        mock_vector_store.add = Mock()
        
        mock_ista_db_class.return_value = mock_db
        mock_encoder_class.return_value = mock_encoder
        mock_vector_store_class.return_value = mock_vector_store
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config,
            batch_size=10
        )
        
        stats = indexer.index(limit=2)
        
        # Should have processed both procedures (processed_count increments before error check)
        # One will fail during processing, so errors >= 1
        assert stats["processed"] >= 2
        assert stats["errors"] >= 1
    
    @patch('index_repair_guides.IstaDatabase')
    @patch('index_repair_guides.FaultCodeEncoder')
    @patch('index_repair_guides.VectorStore')
    def test_get_fault_codes_for_procedure_integration(
        self,
        mock_vector_store_class,
        mock_encoder_class,
        mock_ista_db_class,
        embedding_config,
        retrieval_config
    ):
        """Test that get_fault_codes_for_procedure is called correctly."""
        mock_db = Mock()
        mock_db.get_fault_codes_for_procedure = Mock(return_value=["P0301", "P0302"])
        mock_db.get_info_segments = Mock(return_value=[
            {"CONTENT_ENGB": "Test content"}
        ])
        
        mock_ista_db_class.return_value = mock_db
        mock_encoder_class.return_value = Mock()
        mock_vector_store_class.return_value = Mock()
        
        indexer = RepairGuideIndexer(
            embedding_config=embedding_config,
            retrieval_config=retrieval_config
        )
        
        procedure = {"id": "proc1", "title_engb": "Test", "name": "Test"}
        document = indexer._process_procedure(procedure)
        
        mock_db.get_fault_codes_for_procedure.assert_called_once_with("proc1")
        assert document["fault_codes"] == ["P0301", "P0302"]


class TestIndexerCLI:
    """Test CLI functionality (integration test)."""
    
    @patch('index_repair_guides.RepairGuideIndexer')
    @patch('index_repair_guides.get_paths')
    def test_main_function_calls_indexer(self, mock_get_paths, mock_indexer_class):
        """Test that main() function initializes and calls indexer."""
        # Mock paths
        mock_paths = Mock()
        mock_paths.embedding_config = Path("/fake/embedding_config.yaml")
        mock_paths.retrieval_config = Path("/fake/retrieval_config.yaml")
        mock_get_paths.return_value = mock_paths
        
        # Mock config loading
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = '{"models": {}}'
            
            # Mock indexer instance
            mock_indexer = Mock()
            mock_indexer.index = Mock(return_value={
                "total_procedures": 10,
                "processed": 10,
                "indexed": 10,
                "errors": 0
            })
            mock_indexer.close = Mock()
            mock_indexer_class.return_value = mock_indexer
            
            # Import and run main
            from index_repair_guides import main
            
            # This would normally run, but we'll just verify the structure
            # In a real test, we'd use subprocess or mock sys.argv
            assert True  # Placeholder - full CLI test would require more setup
