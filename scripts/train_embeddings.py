#!/usr/bin/env python3
"""
Train embeddings from feedback data using contrastive learning.

This script loads feedback data, creates training datasets, and fine-tunes
embeddings using the EmbeddingTrainer class with InfoNCE loss.
"""
import sys
import argparse
from pathlib import Path

# Add project root for consistent imports (from src.X)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings.embedding_trainer import EmbeddingTrainer
from src.embeddings.multimodal_encoder import MultiModalEncoder
from src.paths import get_paths
import yaml
import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train embeddings from feedback data using contrastive learning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config file (default: config/training_config.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    parser.add_argument(
        "--embedding-config",
        type=str,
        default=None,
        help="Path to embedding config file (default: config/embedding_config.yaml)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training (default: auto-detect)"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        choices=["db", "feedback", "both"],
        help="Data source: db (scraped_records), feedback (SQLite), or both. Default: db when DATABASE_URL set, else feedback"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Get paths
    paths = get_paths()
    
    try:
        # Load embedding config
        logger.info("Loading embedding configuration...")
        embedding_config_path = Path(args.embedding_config) if args.embedding_config else paths.embedding_config
        if not embedding_config_path.exists():
            raise FileNotFoundError(
                f"Embedding config file not found: {embedding_config_path}"
            )
        
        with open(embedding_config_path, 'r', encoding='utf-8') as f:
            embedding_config = yaml.safe_load(f)
        
        if embedding_config is None:
            raise ValueError(f"Embedding config file is empty: {embedding_config_path}")
        
        # Load training config
        logger.info("Loading training configuration...")
        training_config_path = Path(args.config) if args.config else paths.training_config
        if not training_config_path.exists():
            raise FileNotFoundError(
                f"Training config file not found: {training_config_path}"
            )
        
        with open(training_config_path, 'r', encoding='utf-8') as f:
            training_config = yaml.safe_load(f)
        
        if training_config is None:
            raise ValueError(f"Training config file is empty: {training_config_path}")
        
        # Check device availability
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Initialize MultiModalEncoder with embedding config
        logger.info("Initializing MultiModalEncoder...")
        encoder = MultiModalEncoder(config=embedding_config)
        
        # Initialize EmbeddingTrainer
        logger.info("Initializing EmbeddingTrainer...")
        trainer = EmbeddingTrainer(
            encoder=encoder,
            config=training_config,
            feedback_collector=None  # Will be created internally
        )
        
        # Check if resuming from checkpoint
        resume_from_checkpoint = None
        if args.resume:
            resume_from_checkpoint = Path(args.resume)
            if not resume_from_checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint file not found: {resume_from_checkpoint}"
                )
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        
        # Start training
        logger.info("Starting training...")
        logger.info(f"Training config: {training_config_path}")
        logger.info(f"Embedding config: {embedding_config_path}")
        logger.info(f"Checkpoint directory: {trainer.checkpoint_dir}")
        if args.data_source:
            logger.info(f"Data source: {args.data_source}")
        
        trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
            data_source=args.data_source
        )
        
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Training error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
