"""
Embedding fine-tuning pipeline with contrastive learning.

Supports data_source: db (scraped_records), feedback (SQLite), or both.
When DATABASE_URL is set, db is the default.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from ..learning.losses import InfoNCELoss
from ..feedback.collector import FeedbackCollector
from ..database.schema import FeedbackSession, MistEmbedding
from ..database.connection import create_connection
from ..paths import get_paths

logger = logging.getLogger(__name__)


class ContrastiveFeedbackDataset(Dataset):
    """
    Dataset for contrastive learning from feedback data.
    
    Creates positive pairs (query, selected_guide) and negative pairs
    (query, non-selected guides) for training embeddings.
    """
    
    def __init__(
        self,
        anchors: List[torch.Tensor],
        positives: List[torch.Tensor],
        negatives_list: List[List[torch.Tensor]],
        num_negatives: int = 5
    ):
        """
        Initialize contrastive feedback dataset.
        
        Args:
            anchors: List of anchor embeddings (query embeddings)
            positives: List of positive embeddings (selected guide embeddings)
            negatives_list: List of lists of negative embeddings (non-selected guides)
            num_negatives: Number of negatives to use per anchor (default: 5)
        """
        if len(anchors) != len(positives) or len(anchors) != len(negatives_list):
            raise ValueError(
                f"Mismatch in dataset sizes: anchors={len(anchors)}, "
                f"positives={len(positives)}, negatives_list={len(negatives_list)}"
            )
        
        self.anchors = anchors
        self.positives = positives
        self.negatives_list = negatives_list
        self.num_negatives = num_negatives
    
    def __len__(self) -> int:
        return len(self.anchors)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Tuple of (anchor, positive, negatives) tensors
            - anchor: (embedding_dim,)
            - positive: (embedding_dim,)
            - negatives: (num_negatives, embedding_dim)
        """
        anchor = self.anchors[idx]
        positive = self.positives[idx]
        negatives = self.negatives_list[idx]
        
        # Ensure we have enough negatives
        if len(negatives) < self.num_negatives:
            # Pad with last negative if needed
            while len(negatives) < self.num_negatives:
                negatives.append(negatives[-1] if negatives else anchor)
        elif len(negatives) > self.num_negatives:
            # Sample random negatives
            indices = torch.randperm(len(negatives))[:self.num_negatives]
            negatives = [negatives[i] for i in indices]
        
        # Stack negatives into tensor
        negatives_tensor = torch.stack(negatives[:self.num_negatives])
        
        return anchor, positive, negatives_tensor


class EmbeddingTrainer:
    """
    Trains embeddings using contrastive learning from feedback data.
    
    Uses InfoNCE loss to fine-tune embeddings based on user feedback,
    learning to distinguish between relevant and irrelevant repair guides.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        config: Optional[Union[Dict[str, Any], Path, str]] = None,
        feedback_collector: Optional[FeedbackCollector] = None
    ):
        """
        Initialize embedding trainer.
        
        Args:
            encoder: MultiModalEncoder instance to train
            config: Training configuration dict, Path to config file, or None to load default
            feedback_collector: FeedbackCollector instance. If None, creates one.
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If encoder or device setup fails
        """
        self.encoder = encoder
        self.config = self._load_config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        
        # Initialize feedback collector
        if feedback_collector is None:
            paths = get_paths()
            self.feedback_collector = FeedbackCollector(str(paths.feedback_db))
        else:
            self.feedback_collector = feedback_collector
        
        # Initialize checkpoint directory
        paths = get_paths()
        self.checkpoint_dir = paths.embeddings_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Initialized EmbeddingTrainer on device: {self.device}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _load_config(self, config: Optional[Union[Dict[str, Any], Path, str]]) -> Dict[str, Any]:
        """
        Load configuration from dict, file path, or default location.
        
        Args:
            config: Config dict, Path to YAML file, or None for default
        
        Returns:
            Configuration dictionary with training and fine_tuning sections
        """
        if config is None:
            # Load from default location
            paths = get_paths()
            config_path = paths.training_config
        elif isinstance(config, (str, Path)):
            config_path = Path(config)
        else:
            # Already a dict
            return config
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Failed to parse YAML config file {config_path}: {e}"
            ) from e
        
        if full_config is None:
            raise ValueError(f"Config file {config_path} is empty")
        
        return full_config
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.config.get("training", {})
    
    def _get_fine_tuning_config(self) -> Dict[str, Any]:
        """Get fine-tuning configuration section."""
        return self.config.get("fine_tuning", {})
    
    def _get_guide_embedding(
        self,
        procedure_id: str,
        guide_text: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """
        Get embedding for a guide (procedure).
        
        First tries to load from MistEmbedding table, otherwise encodes guide text.
        
        Args:
            procedure_id: Procedure ID
            guide_text: Optional guide text to encode if embedding not found
        
        Returns:
            Guide embedding tensor or None if not available
        """
        # Try to get from MistEmbedding table
        paths = get_paths()
        mist_db_path = paths.get_mist_db_path()
        
        try:
            connection = create_connection(mist_db_path)
            with connection.session() as session:
                embedding_record = session.query(MistEmbedding).filter_by(
                    procedure_id=procedure_id
                ).first()
                
                if embedding_record and embedding_record.embedding:
                    embedding_np = embedding_record.get_embedding()
                    if embedding_np is not None:
                        # Copy to make writable and convert to tensor
                        return torch.from_numpy(embedding_np.copy()).float()
        except Exception as e:
            logger.debug(f"Could not load embedding from database for {procedure_id}: {e}")
        
        # Fallback: encode guide text if provided
        if guide_text:
            try:
                self.encoder.eval()
                with torch.no_grad():
                    # Encode as fault code (text-based encoding)
                    embedding = self.encoder.encode(guide_text, obd_data=None)
                    return embedding.squeeze(0) if embedding.dim() > 1 else embedding
            except Exception as e:
                logger.debug(f"Could not encode guide text for {procedure_id}: {e}")
        
        return None
    
    def _load_scraped_pairs_from_db(self, db_url: str) -> List[Dict[str, Any]]:
        """Load (fault_codes, repair_summary, matched_guide_id) from scraped_records."""
        from sqlalchemy import create_engine, text
        data_source_config = self.config.get("data_source", {})
        min_confidence = data_source_config.get("scraped_min_confidence", 0.7)
        outcomes = data_source_config.get("scraped_outcomes", ["success", "partial"])
        outcomes_sql = ", ".join(f"'{o}'" for o in outcomes)
        engine = create_engine(db_url)
        pairs = []
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT source_url, fault_codes, repair_summary, matched_guide_id, matched_guide_title, symptoms, record_type
                    FROM scraped_records
                    WHERE outcome IN ({outcomes_sql})
                      AND (confidence_score IS NULL OR confidence_score >= :min_conf)
                      AND repair_summary IS NOT NULL
                      AND (fault_codes IS NOT NULL AND fault_codes::text NOT IN ('[]', 'null')
                           OR record_type = 'cause_to_solution')
                """),
                {"min_conf": min_confidence}
            )
            columns = result.keys()
            for row in result:
                rec = dict(zip(columns, row))
                fault_codes = rec.get("fault_codes")
                if isinstance(fault_codes, str):
                    try:
                        fault_codes = json.loads(fault_codes) if fault_codes else []
                    except json.JSONDecodeError:
                        fault_codes = []
                if not isinstance(fault_codes, list):
                    fault_codes = []
                if not fault_codes and rec.get("record_type") != "cause_to_solution":
                    continue
                pairs.append({
                    "fault_codes": fault_codes,
                    "repair_summary": rec.get("repair_summary") or "",
                    "matched_guide_id": rec.get("matched_guide_id"),
                    "matched_guide_title": rec.get("matched_guide_title"),
                    "source_url": rec.get("source_url"),
                    "symptoms": rec.get("symptoms"),
                    "record_type": rec.get("record_type", "fault_code"),
                })
        return pairs
    
    def create_dataset(
        self,
        min_feedback_samples: Optional[int] = None,
        data_source: Optional[str] = None
    ) -> Tuple[ContrastiveFeedbackDataset, ContrastiveFeedbackDataset]:
        """
        Create training and validation datasets.
        
        Args:
            min_feedback_samples: Minimum number of samples required.
            data_source: "db" | "feedback" | "both". Default: "db" when DATABASE_URL set, else "feedback".
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        fine_tuning_config = self._get_fine_tuning_config()
        data_source_config = self.config.get("data_source", {})
        default_src = data_source_config.get("default", "feedback")
        db_url = os.environ.get("DATABASE_URL", "")
        if data_source is None:
            data_source = "db" if (db_url and db_url.startswith("postgresql")) else default_src
        min_samples = min_feedback_samples or fine_tuning_config.get("min_feedback_samples", 10)
        validation_split = fine_tuning_config.get("validation_split", 0.2)
        
        anchors = []
        positives = []
        negatives_list = []
        
        if data_source in ("db", "both") and db_url and db_url.startswith("postgresql"):
            scraped_pairs = self._load_scraped_pairs_from_db(db_url)
            logger.info(f"Loaded {len(scraped_pairs)} scraped pairs from DB")
            self.encoder.eval()
            with torch.no_grad():
                for i, pair in enumerate(scraped_pairs):
                    try:
                        fault_codes = pair["fault_codes"]
                        repair_summary = pair["repair_summary"]
                        if not repair_summary or (not fault_codes and data_source != "both"):
                            continue
                        fault_codes_str = ", ".join(fault_codes) if fault_codes else ""
                        if not fault_codes_str:
                            fault_codes_str = (pair.get("symptoms") or "")[:500] or "symptoms"
                        anchor = self.encoder.encode(fault_codes_str, obd_data=None)
                        anchor = anchor.squeeze(0) if anchor.dim() > 1 else anchor
                        positive = self._get_guide_embedding(
                            pair.get("matched_guide_id") or "",
                            guide_text=pair.get("matched_guide_title") or repair_summary
                        )
                        if positive is None:
                            positive = self.encoder.encode(repair_summary[:2000], obd_data=None)
                            positive = positive.squeeze(0) if positive.dim() > 1 else positive
                        negs = []
                        my_fc = set(fault_codes or [])
                        for j, other in enumerate(scraped_pairs):
                            if j == i:
                                continue
                            other_fc = set(other.get("fault_codes") or [])
                            if my_fc and other_fc and not my_fc.intersection(other_fc):
                                p = self._get_guide_embedding(
                                    other.get("matched_guide_id") or "",
                                    guide_text=other.get("matched_guide_title") or other.get("repair_summary", "")
                                )
                                if p is None:
                                    p = self.encoder.encode((other.get("repair_summary") or "")[:2000], obd_data=None)
                                    p = p.squeeze(0) if p.dim() > 1 else p
                                negs.append(p)
                                if len(negs) >= 5:
                                    break
                            elif not my_fc and not other_fc:
                                p = self._get_guide_embedding(
                                    other.get("matched_guide_id") or "",
                                    guide_text=other.get("matched_guide_title") or other.get("repair_summary", "")
                                )
                                if p is None:
                                    p = self.encoder.encode((other.get("repair_summary") or "")[:2000], obd_data=None)
                                    p = p.squeeze(0) if p.dim() > 1 else p
                                negs.append(p)
                                if len(negs) >= 5:
                                    break
                        if negs:
                            anchors.append(anchor)
                            positives.append(positive)
                            negatives_list.append(negs)
                    except Exception as e:
                        logger.debug(f"Error processing scraped pair: {e}")
                        continue
            logger.info(f"Created {len(anchors)} samples from scraped DB")
        
        if data_source in ("feedback", "both"):
            session_data_list = []
            try:
                connection = create_connection(self.feedback_collector.db_path)
                with connection.session() as session:
                    sessions = session.query(FeedbackSession).filter(
                        FeedbackSession.selected_guide.isnot(None),
                        FeedbackSession.selected_guide != ""
                    ).all()
                    for feedback_session in sessions:
                        session_data_list.append({
                            "fault_codes": feedback_session.get_fault_codes(),
                            "obd_data": feedback_session.get_obd_data(),
                            "selected_guide": feedback_session.selected_guide,
                            "recommended_guides": feedback_session.get_recommended_guides(),
                            "session_id": feedback_session.session_id
                        })
            except Exception as e:
                raise RuntimeError(f"Failed to query feedback sessions: {e}") from e
            
            if data_source == "feedback" and len(session_data_list) < min_samples:
                raise ValueError(
                    f"Insufficient feedback data: {len(session_data_list)} sessions "
                    f"(minimum {min_samples} required)"
                )
            logger.info(f"Found {len(session_data_list)} feedback sessions with selected guides")
            
            self.encoder.eval()
            with torch.no_grad():
                for session_data in session_data_list:
                    try:
                        fault_codes = session_data["fault_codes"]
                        obd_data = session_data["obd_data"]
                        selected_guide = session_data["selected_guide"]
                        recommended_guides = session_data["recommended_guides"]
                        if not fault_codes or not selected_guide:
                            continue
                        fault_codes_str = ", ".join(fault_codes) if isinstance(fault_codes, list) else str(fault_codes)
                        query_embedding = self.encoder.encode(
                            fault_codes_str,
                            obd_data=obd_data if obd_data else None
                        )
                        query_embedding = query_embedding.squeeze(0) if query_embedding.dim() > 1 else query_embedding
                        positive_embedding = self._get_guide_embedding(selected_guide)
                        if positive_embedding is None:
                            logger.debug(f"Could not get embedding for selected guide: {selected_guide}")
                            continue
                        negative_embeddings = []
                        for guide_id in recommended_guides:
                            if guide_id != selected_guide:
                                neg_emb = self._get_guide_embedding(guide_id)
                                if neg_emb is not None:
                                    negative_embeddings.append(neg_emb)
                        if negative_embeddings:
                            anchors.append(query_embedding)
                            positives.append(positive_embedding)
                            negatives_list.append(negative_embeddings)
                    except Exception as e:
                        logger.warning(f"Error processing session {session_data.get('session_id', 'unknown')}: {e}")
                        continue
        
        if len(anchors) < min_samples:
            raise ValueError(
                f"Insufficient valid samples after processing: {len(anchors)} "
                f"(minimum {min_samples} required)"
            )
        
        logger.info(f"Created {len(anchors)} training samples")
        
        # Split into train/val
        if validation_split > 0 and len(anchors) > 1:
            train_indices, val_indices = train_test_split(
                range(len(anchors)),
                test_size=validation_split,
                random_state=42,
                shuffle=True
            )
            
            train_anchors = [anchors[i] for i in train_indices]
            train_positives = [positives[i] for i in train_indices]
            train_negatives = [negatives_list[i] for i in train_indices]
            
            val_anchors = [anchors[i] for i in val_indices]
            val_positives = [positives[i] for i in val_indices]
            val_negatives = [negatives_list[i] for i in val_indices]
            
            train_dataset = ContrastiveFeedbackDataset(
                train_anchors, train_positives, train_negatives
            )
            val_dataset = ContrastiveFeedbackDataset(
                val_anchors, val_positives, val_negatives
            )
            
            logger.info(
                f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val"
            )
        else:
            # No validation split
            train_dataset = ContrastiveFeedbackDataset(
                anchors, positives, negatives_list
            )
            val_dataset = ContrastiveFeedbackDataset([], [], [])
            logger.info(f"Using all {len(train_dataset)} samples for training")
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_dataset: Optional[ContrastiveFeedbackDataset] = None,
        val_dataset: Optional[ContrastiveFeedbackDataset] = None,
        resume_from_checkpoint: Optional[Union[str, Path]] = None,
        data_source: Optional[str] = None
    ) -> None:
        """
        Train encoder using contrastive learning.
        
        Args:
            train_dataset: Training dataset. If None, creates from data_source.
            val_dataset: Validation dataset. If None, creates from data_source.
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            data_source: "db" | "feedback" | "both". Used when creating dataset.
        
        Raises:
            ValueError: If datasets are invalid or insufficient data
            RuntimeError: If training fails
        """
        # Create datasets if not provided
        if train_dataset is None or val_dataset is None:
            train_dataset, val_dataset = self.create_dataset(data_source=data_source)
        
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        
        # Load checkpoint if resuming
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Get training config
        training_config = self._get_training_config()
        fine_tuning_config = self._get_fine_tuning_config()
        
        batch_size = training_config.get("batch_size", 32)
        learning_rate = training_config.get("learning_rate", 1e-5)
        num_epochs = training_config.get("num_epochs", 10)
        warmup_steps = training_config.get("warmup_steps", 100)
        weight_decay = training_config.get("weight_decay", 0.01)
        temperature = training_config.get("temperature", 0.05)
        gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        checkpoint_interval = fine_tuning_config.get("checkpoint_interval", 1)
        early_stopping_patience = fine_tuning_config.get("early_stopping_patience", 3)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler (linear warmup + cosine decay)
        total_steps = len(train_dataset) // batch_size * num_epochs
        scheduler = self._create_scheduler(optimizer, warmup_steps, total_steps)
        
        # Initialize loss function
        loss_fn = InfoNCELoss(temperature=temperature, reduction='mean')
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = None
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available()
            )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                loss_fn,
                gradient_accumulation_steps
            )
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, loss_fn)
            
            # Logging
            log_msg = (
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}"
            )
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.4f}"
            log_msg += f", LR: {scheduler.get_last_lr()[0]:.2e}"
            logger.info(log_msg)
            
            # Checkpointing
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss, optimizer, scheduler)
                
                # Also save as latest
                latest_path = self.checkpoint_dir / "latest.pt"
                self.save_checkpoint(latest_path, epoch, train_loss, val_loss, optimizer, scheduler)
            
            # Early stopping
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    best_path = self.checkpoint_dir / "best.pt"
                    self.save_checkpoint(best_path, epoch, train_loss, val_loss, optimizer, scheduler)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {epoch+1} epochs "
                            f"(patience: {early_stopping_patience})"
                        )
                        break
        
        logger.info("Training completed")
    
    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler with warmup and cosine decay.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
        
        Returns:
            Learning rate scheduler
        """
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: InfoNCELoss,
        gradient_accumulation_steps: int
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            gradient_accumulation_steps: Steps to accumulate gradients
        
        Returns:
            Average training loss
        """
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (anchors, positives, negatives) in enumerate(train_loader):
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)  # (batch_size, num_negatives, embedding_dim)
            
            # Forward pass
            loss = loss_fn(anchors, positives, negatives)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
        
        # Handle remaining gradients
        if num_batches % gradient_accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        loss_fn: InfoNCELoss
    ) -> float:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
        
        Returns:
            Average validation loss
        """
        self.encoder.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for anchors, positives, negatives in val_loader:
                anchors = anchors.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)
                
                loss = loss_fn(anchors, positives, negatives)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            optimizer: Optimizer state
            scheduler: Scheduler state
        """
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(
        self,
        path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If loading fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.current_epoch = checkpoint.get("epoch", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            self.patience_counter = checkpoint.get("patience_counter", 0)
            
            logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
            return checkpoint
        
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}") from e
