"""Train embeddings from feedback data using contrastive learning."""
import sys
import logging
from pathlib import Path

import torch
import yaml

from src.embeddings.embedding_trainer import EmbeddingTrainer
from src.embeddings.multimodal_encoder import MultiModalEncoder
from src.paths import get_paths

logger = logging.getLogger(__name__)


def run(
    config: str | None = None,
    resume: str | None = None,
    embedding_config: str | None = None,
    log_level: str = "INFO",
    device: str = "auto",
    data_source: str | None = None,
) -> int:
    """Run training. Returns 0 on success."""
    logging.getLogger().setLevel(getattr(logging, log_level))
    paths = get_paths()

    try:
        embedding_config_path = Path(embedding_config) if embedding_config else paths.embedding_config
        if not embedding_config_path.exists():
            raise FileNotFoundError(f"Embedding config file not found: {embedding_config_path}")

        with open(embedding_config_path, "r", encoding="utf-8") as f:
            embedding_config_data = yaml.safe_load(f)
        if embedding_config_data is None:
            raise ValueError(f"Embedding config file is empty: {embedding_config_path}")

        training_config_path = Path(config) if config else paths.training_config
        if not training_config_path.exists():
            raise FileNotFoundError(f"Training config file not found: {training_config_path}")

        with open(training_config_path, "r", encoding="utf-8") as f:
            training_config = yaml.safe_load(f)
        if training_config is None:
            raise ValueError(f"Training config file is empty: {training_config_path}")

        if device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device_str = "cpu"
        logger.info("Using device: %s", device_str)

        logger.info("Initializing MultiModalEncoder...")
        encoder = MultiModalEncoder(config=embedding_config_data)

        logger.info("Initializing EmbeddingTrainer...")
        trainer = EmbeddingTrainer(
            encoder=encoder,
            config=training_config,
            feedback_collector=None,
        )

        resume_from_checkpoint = None
        if resume:
            resume_from_checkpoint = Path(resume)
            if not resume_from_checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_from_checkpoint}")
            logger.info("Resuming training from checkpoint: %s", resume_from_checkpoint)

        logger.info("Starting training...")
        logger.info("Training config: %s", training_config_path)
        logger.info("Embedding config: %s", embedding_config_path)
        logger.info("Checkpoint directory: %s", trainer.checkpoint_dir)
        if data_source:
            logger.info("Data source: %s", data_source)

        trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
            data_source=data_source,
        )

        logger.info("Training completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except (RuntimeError, Exception) as e:
        logger.error("Training error: %s", e, exc_info=True)
        return 1
