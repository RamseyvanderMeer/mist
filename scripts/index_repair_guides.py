#!/usr/bin/env python3
"""
Index repair guides in vector store.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.multimodal_encoder import MultiModalEncoder
from retrieval.vector_store import VectorStore
from paths import get_paths
import yaml
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """Index repair guides"""
    paths = get_paths()
    
    # Load configs
    with open(paths.config / "embedding_config.yaml") as f:
        embedding_config = yaml.safe_load(f)
    
    with open(paths.config / "retrieval_config.yaml") as f:
        retrieval_config = yaml.safe_load(f)
    
    # Initialize encoder and vector store
    encoder = MultiModalEncoder()
    vector_store = VectorStore(retrieval_config["vector_store"])
    
    # TODO: Load repair guides from database and index them
    print("Indexing repair guides...")
    print("TODO: Implement database query and indexing")

if __name__ == "__main__":
    main()
