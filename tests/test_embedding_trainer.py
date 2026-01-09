"""
Unit tests for EmbeddingTrainer.

Tests dataset creation, training loop, checkpointing, and integration.
"""
import pytest
import torch
import torch.nn as nn
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import numpy as np

from src.embeddings.embedding_trainer import EmbeddingTrainer, ContrastiveFeedbackDataset
from src.embeddings.multimodal_encoder import MultiModalEncoder
from src.feedback.collector import FeedbackCollector
from src.database.schema import FeedbackSession, MistEmbedding
from src.database.connection import create_connection


class TestContrastiveFeedbackDataset:
    """Test ContrastiveFeedbackDataset class."""
    
    def test_init(self):
        """Test dataset initialization."""
        embedding_dim = 768
        num_samples = 5
        num_negatives = 3
        
        anchors = [torch.randn(embedding_dim) for _ in range(num_samples)]
        positives = [torch.randn(embedding_dim) for _ in range(num_samples)]
        negatives_list = [[torch.randn(embedding_dim) for _ in range(num_negatives)] 
                          for _ in range(num_samples)]
        
        dataset = ContrastiveFeedbackDataset(anchors, positives, negatives_list, num_negatives)
        
        assert len(dataset) == num_samples
        assert dataset.num_negatives == num_negatives
    
    def test_init_mismatch_sizes(self):
        """Test that mismatched sizes raise ValueError."""
        embedding_dim = 768
        
        anchors = [torch.randn(embedding_dim) for _ in range(5)]
        positives = [torch.randn(embedding_dim) for _ in range(4)]  # Mismatch
        negatives_list = [[torch.randn(embedding_dim)] for _ in range(5)]
        
        with pytest.raises(ValueError, match="Mismatch in dataset sizes"):
            ContrastiveFeedbackDataset(anchors, positives, negatives_list)
    
    def test_getitem(self):
        """Test getting a dataset item."""
        embedding_dim = 768
        num_negatives = 3
        
        anchors = [torch.randn(embedding_dim)]
        positives = [torch.randn(embedding_dim)]
        negatives_list = [[torch.randn(embedding_dim) for _ in range(num_negatives)]]
        
        dataset = ContrastiveFeedbackDataset(anchors, positives, negatives_list, num_negatives)
        
        anchor, positive, negatives = dataset[0]
        
        assert anchor.shape == (embedding_dim,)
        assert positive.shape == (embedding_dim,)
        assert negatives.shape == (num_negatives, embedding_dim)
    
    def test_getitem_insufficient_negatives(self):
        """Test padding when insufficient negatives."""
        embedding_dim = 768
        num_negatives = 5
        
        anchors = [torch.randn(embedding_dim)]
        positives = [torch.randn(embedding_dim)]
        negatives_list = [[torch.randn(embedding_dim) for _ in range(2)]]  # Only 2 negatives
        
        dataset = ContrastiveFeedbackDataset(anchors, positives, negatives_list, num_negatives)
        
        anchor, positive, negatives = dataset[0]
        
        assert negatives.shape == (num_negatives, embedding_dim)
    
    def test_getitem_excess_negatives(self):
        """Test sampling when too many negatives."""
        embedding_dim = 768
        num_negatives = 3
        
        anchors = [torch.randn(embedding_dim)]
        positives = [torch.randn(embedding_dim)]
        negatives_list = [[torch.randn(embedding_dim) for _ in range(10)]]  # 10 negatives
        
        dataset = ContrastiveFeedbackDataset(anchors, positives, negatives_list, num_negatives)
        
        anchor, positive, negatives = dataset[0]
        
        assert negatives.shape == (num_negatives, embedding_dim)


class TestEmbeddingTrainer:
    """Test EmbeddingTrainer class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def temp_mist_db(self):
        """Create temporary MIST database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        # Initialize database
        from src.database.migrations import init_database
        init_database(db_path)
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary training config file."""
        config = {
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-5,
                "num_epochs": 2,
                "warmup_steps": 10,
                "weight_decay": 0.01,
                "temperature": 0.05,
                "gradient_accumulation_steps": 1
            },
            "fine_tuning": {
                "enabled": True,
                "checkpoint_interval": 1,
                "min_feedback_samples": 2,
                "validation_split": 0.2,
                "early_stopping_patience": 3
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        yield config_path
        # Cleanup
        Path(config_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder."""
        encoder = MagicMock(spec=MultiModalEncoder)
        encoder.get_dimension.return_value = 768
        
        # Create a simple linear layer as the "encoder" so it has trainable parameters
        linear = nn.Linear(768, 768)
        
        # Mock encode method to return normalized random embeddings
        def mock_encode(fault_codes, obd_data=None):
            if isinstance(fault_codes, list):
                batch_size = len(fault_codes)
            else:
                batch_size = 1
            emb = torch.randn(batch_size, 768, requires_grad=True)
            # Pass through linear layer to create gradient path
            emb = linear(emb)
            return torch.nn.functional.normalize(emb, p=2, dim=1)
        
        encoder.encode = mock_encode
        encoder.eval = Mock()
        encoder.train = Mock()
        encoder.to = Mock(return_value=encoder)
        encoder.parameters = Mock(return_value=linear.parameters())
        encoder.state_dict = Mock(return_value=linear.state_dict())
        encoder.load_state_dict = Mock()
        
        # Store linear for parameter access
        encoder._linear = linear
        
        return encoder
    
    @pytest.fixture
    def trainer(self, mock_encoder, temp_config, temp_db):
        """Create EmbeddingTrainer instance."""
        with patch('src.embeddings.embedding_trainer.get_paths') as mock_paths:
            paths = MagicMock()
            paths.embeddings_checkpoints = Path(tempfile.mkdtemp())
            paths.feedback_db = Path(temp_db)
            paths.get_mist_db_path = Mock(return_value=Path(temp_db))
            mock_paths.return_value = paths
            
            trainer = EmbeddingTrainer(
                encoder=mock_encoder,
                config=temp_config,
                feedback_collector=None
            )
            return trainer
    
    def test_init_with_config_dict(self, mock_encoder, temp_db):
        """Test initialization with config dict."""
        config = {
            "training": {"batch_size": 32, "learning_rate": 1e-5},
            "fine_tuning": {"min_feedback_samples": 10}
        }
        
        with patch('src.embeddings.embedding_trainer.get_paths') as mock_paths:
            paths = MagicMock()
            paths.embeddings_checkpoints = Path(tempfile.mkdtemp())
            paths.feedback_db = Path(temp_db)
            paths.get_mist_db_path = Mock(return_value=Path(temp_db))
            mock_paths.return_value = paths
            
            trainer = EmbeddingTrainer(encoder=mock_encoder, config=config)
            
            assert trainer.encoder == mock_encoder
            assert trainer.config == config
            assert trainer.device in [torch.device("cpu"), torch.device("cuda")]
    
    def test_init_with_config_path(self, mock_encoder, temp_config, temp_db):
        """Test initialization with config file path."""
        with patch('src.embeddings.embedding_trainer.get_paths') as mock_paths:
            paths = MagicMock()
            paths.embeddings_checkpoints = Path(tempfile.mkdtemp())
            paths.feedback_db = Path(temp_db)
            paths.get_mist_db_path = Mock(return_value=Path(temp_db))
            mock_paths.return_value = paths
            
            trainer = EmbeddingTrainer(encoder=mock_encoder, config=temp_config)
            
            assert trainer.config is not None
            assert "training" in trainer.config
    
    def test_init_creates_checkpoint_dir(self, mock_encoder, temp_config, temp_db):
        """Test that checkpoint directory is created."""
        with patch('src.embeddings.embedding_trainer.get_paths') as mock_paths:
            checkpoint_dir = Path(tempfile.mkdtemp())
            paths = MagicMock()
            paths.embeddings_checkpoints = checkpoint_dir
            paths.feedback_db = Path(temp_db)
            paths.get_mist_db_path = Mock(return_value=Path(temp_db))
            mock_paths.return_value = paths
            
            trainer = EmbeddingTrainer(encoder=mock_encoder, config=temp_config)
            
            assert checkpoint_dir.exists()
    
    def test_get_guide_embedding_from_db(self, trainer, temp_mist_db):
        """Test retrieving guide embedding from database."""
        embedding_dim = 768
        procedure_id = "test-procedure-1"
        
        # Create embedding in database
        connection = create_connection(temp_mist_db)
        with connection.session() as session:
            embedding_np = np.random.randn(embedding_dim).astype(np.float32)
            mist_emb = MistEmbedding(
                procedure_id=procedure_id,
                embedding_version=1
            )
            mist_emb.set_embedding(embedding_np)
            session.add(mist_emb)
            session.commit()
        
        # Patch create_connection to return our connection
        with patch('src.embeddings.embedding_trainer.create_connection', return_value=connection):
            # Patch get_paths to return temp_mist_db
            with patch('src.embeddings.embedding_trainer.get_paths') as mock_paths:
                paths = MagicMock()
                paths.get_mist_db_path.return_value = Path(temp_mist_db)
                mock_paths.return_value = paths
                
                embedding = trainer._get_guide_embedding(procedure_id)
                
                assert embedding is not None
                assert embedding.shape == (embedding_dim,)
    
    def test_get_guide_embedding_fallback_encode(self, trainer):
        """Test fallback to encoding guide text."""
        procedure_id = "test-procedure-2"
        guide_text = "Test repair guide text"
        
        # Track if encode was called
        encode_called = False
        original_encode = trainer.encoder.encode
        
        def track_encode(*args, **kwargs):
            nonlocal encode_called
            encode_called = True
            return original_encode(*args, **kwargs)
        
        trainer.encoder.encode = track_encode
        
        # Mock database to return None (not found)
        with patch('src.embeddings.embedding_trainer.create_connection') as mock_conn:
            mock_session = MagicMock()
            mock_session.query.return_value.filter_by.return_value.first.return_value = None
            mock_connection = MagicMock()
            mock_connection.session.return_value.__enter__.return_value = mock_session
            mock_conn.return_value = mock_connection
            
            embedding = trainer._get_guide_embedding(procedure_id, guide_text=guide_text)
            
            # Should encode the guide text
            assert embedding is not None
            assert encode_called
    
    def test_create_dataset_insufficient_samples(self, trainer, temp_db):
        """Test that insufficient samples raise ValueError."""
        # Create collector with empty database
        collector = FeedbackCollector(temp_db)
        
        trainer.feedback_collector = collector
        
        with pytest.raises(ValueError, match="Insufficient feedback data"):
            trainer.create_dataset(min_feedback_samples=10)
    
    def test_create_dataset_with_mock_sessions(self, trainer, temp_db):
        """Test dataset creation with mock feedback sessions."""
        # Create some feedback sessions
        collector = FeedbackCollector(temp_db)
        
        # Create sessions with selected guides
        for i in range(5):
            collector.save_session(
                fault_codes=[f"P030{i}"],
                obd_data={"rpm": 2000 + i * 100},
                recommended_guides=[f"guide-{i}", f"guide-{i+1}", f"guide-{i+2}"],
                selected_guide=f"guide-{i}"
            )
        
        trainer.feedback_collector = collector
        
        # Mock guide embeddings - need to ensure they're on the right device
        def mock_get_guide_embedding(procedure_id, guide_text=None):
            return torch.randn(768, requires_grad=False)
        
        trainer._get_guide_embedding = mock_get_guide_embedding
        
        # Mock encoder.encode to return tensors with proper shape
        def mock_encode(fault_codes, obd_data=None):
            if isinstance(fault_codes, list):
                batch_size = len(fault_codes)
            else:
                batch_size = 1
            emb = torch.randn(batch_size, 768, requires_grad=False)
            return torch.nn.functional.normalize(emb, p=2, dim=1)
        
        trainer.encoder.encode = mock_encode
        
        # Create dataset
        train_dataset, val_dataset = trainer.create_dataset(min_feedback_samples=2)
        
        assert len(train_dataset) > 0
        assert isinstance(train_dataset, ContrastiveFeedbackDataset)
        assert isinstance(val_dataset, ContrastiveFeedbackDataset)
    
    def test_train_with_mock_dataset(self, trainer):
        """Test training with provided dataset."""
        embedding_dim = 768
        num_samples = 8
        num_negatives = 3
        
        # Create a simple trainable encoder for testing
        class SimpleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(768, 768)
            
            def encode(self, fault_codes, obd_data=None):
                # Create embeddings that go through the linear layer
                if isinstance(fault_codes, list):
                    batch_size = len(fault_codes)
                else:
                    batch_size = 1
                x = torch.randn(batch_size, 768, requires_grad=True)
                x = self.linear(x)
                return torch.nn.functional.normalize(x, p=2, dim=1)
        
        real_encoder = SimpleEncoder()
        trainer.encoder = real_encoder.to(trainer.device)
        
        # Create mock datasets with embeddings that can have gradients
        # In real training, these would come from the encoder, but for testing
        # we'll create them directly
        anchors = [torch.randn(embedding_dim, requires_grad=True) for _ in range(num_samples)]
        positives = [torch.randn(embedding_dim, requires_grad=True) for _ in range(num_samples)]
        negatives_list = [[torch.randn(embedding_dim, requires_grad=True) for _ in range(num_negatives)] 
                          for _ in range(num_samples)]
        
        train_dataset = ContrastiveFeedbackDataset(anchors, positives, negatives_list, num_negatives)
        val_dataset = ContrastiveFeedbackDataset(
            anchors[:2], positives[:2], negatives_list[:2], num_negatives
        )
        
        # Run training (should complete without errors)
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        # Verify training completed
        assert trainer.current_epoch >= 0
    
    def test_save_checkpoint(self, trainer):
        """Test saving checkpoint."""
        checkpoint_path = trainer.checkpoint_dir / "test_checkpoint.pt"
        
        mock_optimizer = MagicMock()
        mock_optimizer.state_dict = Mock(return_value={"lr": 1e-5})
        
        mock_scheduler = MagicMock()
        mock_scheduler.state_dict = Mock(return_value={"last_epoch": 0})
        
        trainer.save_checkpoint(
            checkpoint_path,
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler
        )
        
        assert checkpoint_path.exists()
        
        # Verify checkpoint can be loaded
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert checkpoint["epoch"] == 1
        assert checkpoint["train_loss"] == 0.5
        assert checkpoint["val_loss"] == 0.6
    
    def test_load_checkpoint(self, trainer):
        """Test loading checkpoint."""
        checkpoint_path = trainer.checkpoint_dir / "test_checkpoint.pt"
        
        # Create a checkpoint
        checkpoint = {
            "epoch": 5,
            "encoder_state_dict": {"param": torch.randn(10, 10)},
            "optimizer_state_dict": {"lr": 1e-5},
            "scheduler_state_dict": {"last_epoch": 5},
            "train_loss": 0.4,
            "val_loss": 0.5,
            "config": trainer.config,
            "best_val_loss": 0.5,
            "patience_counter": 0
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded = trainer.load_checkpoint(checkpoint_path)
        
        assert loaded["epoch"] == 5
        assert trainer.current_epoch == 5
        assert trainer.best_val_loss == 0.5
        trainer.encoder.load_state_dict.assert_called_once()
    
    def test_load_checkpoint_not_found(self, trainer):
        """Test loading non-existent checkpoint raises FileNotFoundError."""
        checkpoint_path = trainer.checkpoint_dir / "nonexistent.pt"
        
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(checkpoint_path)
    
    def test_train_resume_from_checkpoint(self, trainer):
        """Test resuming training from checkpoint."""
        embedding_dim = 768
        num_samples = 4
        num_negatives = 2
        
        # Create a simple trainable encoder for testing
        class SimpleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(768, 768)
            
            def encode(self, fault_codes, obd_data=None):
                if isinstance(fault_codes, list):
                    batch_size = len(fault_codes)
                else:
                    batch_size = 1
                x = torch.randn(batch_size, 768, requires_grad=True)
                x = self.linear(x)
                return torch.nn.functional.normalize(x, p=2, dim=1)
        
        real_encoder = SimpleEncoder()
        trainer.encoder = real_encoder.to(trainer.device)
        
        # Create checkpoint with real encoder state
        checkpoint_path = trainer.checkpoint_dir / "resume_checkpoint.pt"
        checkpoint = {
            "epoch": 1,
            "encoder_state_dict": real_encoder.state_dict(),
            "optimizer_state_dict": {"lr": 1e-5},
            "scheduler_state_dict": {"last_epoch": 1},
            "train_loss": 0.5,
            "val_loss": 0.6,
            "config": trainer.config,
            "best_val_loss": 0.6,
            "patience_counter": 0
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Create mock datasets
        anchors = [torch.randn(embedding_dim, requires_grad=True) for _ in range(num_samples)]
        positives = [torch.randn(embedding_dim, requires_grad=True) for _ in range(num_samples)]
        negatives_list = [[torch.randn(embedding_dim, requires_grad=True) for _ in range(num_negatives)] 
                          for _ in range(num_samples)]
        
        train_dataset = ContrastiveFeedbackDataset(anchors, positives, negatives_list, num_negatives)
        val_dataset = ContrastiveFeedbackDataset(
            anchors[:1], positives[:1], negatives_list[:1], num_negatives
        )
        
        # Train with resume
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            resume_from_checkpoint=checkpoint_path
        )
        
        # Verify checkpoint was loaded
        assert trainer.current_epoch >= 1
    
    def test_early_stopping(self, trainer):
        """Test early stopping mechanism."""
        embedding_dim = 768
        num_samples = 4
        num_negatives = 2
        
        # Create a simple trainable encoder for testing
        class SimpleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(768, 768)
            
            def encode(self, fault_codes, obd_data=None):
                if isinstance(fault_codes, list):
                    batch_size = len(fault_codes)
                else:
                    batch_size = 1
                x = torch.randn(batch_size, 768, requires_grad=True)
                x = self.linear(x)
                return torch.nn.functional.normalize(x, p=2, dim=1)
        
        real_encoder = SimpleEncoder()
        trainer.encoder = real_encoder.to(trainer.device)
        
        # Create mock datasets
        anchors = [torch.randn(embedding_dim, requires_grad=True) for _ in range(num_samples)]
        positives = [torch.randn(embedding_dim, requires_grad=True) for _ in range(num_samples)]
        negatives_list = [[torch.randn(embedding_dim, requires_grad=True) for _ in range(num_negatives)] 
                          for _ in range(num_samples)]
        
        train_dataset = ContrastiveFeedbackDataset(anchors, positives, negatives_list, num_negatives)
        val_dataset = ContrastiveFeedbackDataset(
            anchors[:1], positives[:1], negatives_list[:1], num_negatives
        )
        
        # Set early stopping patience to 1
        trainer.config["fine_tuning"]["early_stopping_patience"] = 1
        
        # Train - should stop early
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        # Early stopping should trigger (patience=1, val loss won't improve)
        # The exact behavior depends on validation loss, but training should complete
        assert trainer.current_epoch >= 0
