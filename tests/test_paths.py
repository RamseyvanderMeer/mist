"""
Unit tests for path management system.

Tests cover:
- All path properties return correct Path objects
- Environment variable overrides work correctly
- Path resolution on different platforms
- Directory creation functionality
- Config file path methods
- Convenience methods
"""
import os
import tempfile
from pathlib import Path
import pytest

from src.paths import Paths, get_paths


class TestPathsBasic:
    """Test basic path properties."""
    
    def test_mist_root_auto_detection(self):
        """Test that mist_root is auto-detected correctly."""
        paths = Paths()
        # Should resolve to the parent of src directory
        assert paths.mist_root.exists()
        assert paths.mist_root.is_dir()
        assert (paths.mist_root / "src").exists()
    
    def test_mist_root_custom(self):
        """Test custom mist_root path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_root = Path(tmpdir)
            paths = Paths(mist_root=custom_root)
            assert paths.mist_root == custom_root.resolve()
    
    def test_config_directory(self):
        """Test config directory path."""
        paths = Paths()
        config_path = paths.config
        assert isinstance(config_path, Path)
        assert config_path.name == "config"
        assert config_path.parent == paths.mist_root
    
    def test_data_directory(self):
        """Test data directory path."""
        paths = Paths()
        data_path = paths.data
        assert isinstance(data_path, Path)
        assert data_path.name == "data"
        assert data_path.parent == paths.mist_root
    
    def test_src_directory(self):
        """Test src directory path."""
        paths = Paths()
        src_path = paths.src
        assert isinstance(src_path, Path)
        assert src_path.name == "src"
        assert src_path.parent == paths.mist_root
    
    def test_scripts_directory(self):
        """Test scripts directory path."""
        paths = Paths()
        scripts_path = paths.scripts
        assert isinstance(scripts_path, Path)
        assert scripts_path.name == "scripts"
        assert scripts_path.parent == paths.mist_root
    
    def test_migrations_directory(self):
        """Test migrations directory path."""
        paths = Paths()
        migrations_path = paths.migrations
        assert isinstance(migrations_path, Path)
        assert migrations_path.name == "migrations"
        assert migrations_path.parent == paths.scripts
    
    def test_tests_directory(self):
        """Test tests directory path."""
        paths = Paths()
        tests_path = paths.tests
        assert isinstance(tests_path, Path)
        assert tests_path.name == "tests"
        assert tests_path.parent == paths.mist_root
    
    def test_databases_directory(self):
        """Test databases directory path."""
        paths = Paths()
        databases_path = paths.databases
        assert isinstance(databases_path, Path)
        assert databases_path.parent == paths.data
        assert databases_path.name == "databases"
    
    def test_knowledge_graph_path(self):
        """Test knowledge graph file path."""
        paths = Paths()
        kg_path = paths.knowledge_graph
        assert isinstance(kg_path, Path)
        assert kg_path.name == "knowledge_graph.graphml"
        assert kg_path.parent == paths.data
    
    def test_vector_store_directory(self):
        """Test vector store directory path."""
        paths = Paths()
        vs_path = paths.vector_store
        assert isinstance(vs_path, Path)
        assert vs_path.name == "vector_store"
        assert vs_path.parent == paths.data
    
    def test_feedback_directory(self):
        """Test feedback directory path."""
        paths = Paths()
        feedback_path = paths.feedback
        assert isinstance(feedback_path, Path)
        assert feedback_path.name == "feedback"
        assert feedback_path.parent == paths.data
    
    def test_feedback_db_path(self):
        """Test feedback database file path."""
        paths = Paths()
        feedback_db_path = paths.feedback_db
        assert isinstance(feedback_db_path, Path)
        assert feedback_db_path.name == "feedback.db"
        assert feedback_db_path.parent == paths.feedback
    
    def test_embeddings_directory(self):
        """Test embeddings directory path."""
        paths = Paths()
        embeddings_path = paths.embeddings
        assert isinstance(embeddings_path, Path)
        assert embeddings_path.name == "embeddings"
        assert embeddings_path.parent == paths.data
    
    def test_embeddings_checkpoints_directory(self):
        """Test embeddings checkpoints directory path."""
        paths = Paths()
        checkpoints_path = paths.embeddings_checkpoints
        assert isinstance(checkpoints_path, Path)
        assert checkpoints_path.name == "checkpoints"
        assert checkpoints_path.parent == paths.embeddings


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""
    
    def test_config_dir_override(self):
        """Test MIST_CONFIG_DIR environment variable override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MIST_CONFIG_DIR"] = tmpdir
            try:
                paths = Paths()
                assert paths.config == Path(tmpdir).resolve()
            finally:
                del os.environ["MIST_CONFIG_DIR"]
    
    def test_scripts_dir_override(self):
        """Test MIST_SCRIPTS_DIR environment variable override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MIST_SCRIPTS_DIR"] = tmpdir
            try:
                paths = Paths()
                assert paths.scripts == Path(tmpdir).resolve()
            finally:
                del os.environ["MIST_SCRIPTS_DIR"]
    
    def test_database_dir_override(self):
        """Test MIST_DATABASE_DIR environment variable override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MIST_DATABASE_DIR"] = tmpdir
            try:
                paths = Paths()
                assert paths.databases == Path(tmpdir).resolve()
            finally:
                del os.environ["MIST_DATABASE_DIR"]
    
    def test_vector_store_dir_override(self):
        """Test MIST_VECTOR_STORE_DIR environment variable override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MIST_VECTOR_STORE_DIR"] = tmpdir
            try:
                paths = Paths()
                assert paths.vector_store == Path(tmpdir).resolve()
            finally:
                del os.environ["MIST_VECTOR_STORE_DIR"]
    
    def test_env_override_does_not_affect_other_paths(self):
        """Test that overriding one path doesn't affect others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MIST_CONFIG_DIR"] = tmpdir
            try:
                paths = Paths()
                # Config should be overridden
                assert paths.config == Path(tmpdir).resolve()
                # But data should still be relative to mist_root
                assert paths.data.parent == paths.mist_root
            finally:
                del os.environ["MIST_CONFIG_DIR"]


class TestConfigFilePaths:
    """Test config file path methods."""
    
    def test_get_config_path(self):
        """Test get_config_path method."""
        paths = Paths()
        config_path = paths.get_config_path("test_config.yaml")
        assert isinstance(config_path, Path)
        assert config_path.name == "test_config.yaml"
        assert config_path.parent == paths.config
    
    def test_embedding_config_property(self):
        """Test embedding_config property."""
        paths = Paths()
        config_path = paths.embedding_config
        assert isinstance(config_path, Path)
        assert config_path.name == "embedding_config.yaml"
        assert config_path.parent == paths.config
    
    def test_llm_config_property(self):
        """Test llm_config property."""
        paths = Paths()
        config_path = paths.llm_config
        assert isinstance(config_path, Path)
        assert config_path.name == "llm_config.yaml"
        assert config_path.parent == paths.config
    
    def test_retrieval_config_property(self):
        """Test retrieval_config property."""
        paths = Paths()
        config_path = paths.retrieval_config
        assert isinstance(config_path, Path)
        assert config_path.name == "retrieval_config.yaml"
        assert config_path.parent == paths.config
    
    def test_training_config_property(self):
        """Test training_config property."""
        paths = Paths()
        config_path = paths.training_config
        assert isinstance(config_path, Path)
        assert config_path.name == "training_config.yaml"
        assert config_path.parent == paths.config


class TestConvenienceMethods:
    """Test convenience methods."""
    
    def test_get_database_path(self):
        """Test get_database_path method."""
        paths = Paths()
        db_path = paths.get_database_path("test.db")
        assert isinstance(db_path, Path)
        assert db_path.name == "test.db"
        assert db_path.parent == paths.databases
    
    def test_get_mist_db_path(self):
        """Test get_mist_db_path method."""
        paths = Paths()
        mist_db_path = paths.get_mist_db_path()
        assert isinstance(mist_db_path, Path)
        assert mist_db_path.name == "mist_data.db"
        assert mist_db_path.parent == paths.databases
    
    def test_get_migration_sql_path(self):
        """Test get_migration_sql_path method."""
        paths = Paths()
        migration_path = paths.get_migration_sql_path()
        assert isinstance(migration_path, Path)
        assert migration_path.name == "create_mist_tables.sql"
        assert migration_path.parent == paths.migrations


class TestDirectoryCreation:
    """Test directory creation functionality."""
    
    def test_ensure_directories_creates_missing(self):
        """Test that ensure_directories creates missing directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_root = Path(tmpdir)
            paths = Paths(mist_root=custom_root)
            
            # Ensure directories are created
            paths.ensure_directories(create_if_missing=True)
            
            # Verify all directories exist
            assert paths.config.exists()
            assert paths.data.exists()
            assert paths.src.exists()
            assert paths.scripts.exists()
            assert paths.migrations.exists()
            assert paths.tests.exists()
            assert paths.databases.exists()
            assert paths.vector_store.exists()
            assert paths.feedback.exists()
            assert paths.embeddings.exists()
            assert paths.embeddings_checkpoints.exists()
    
    def test_ensure_directories_no_create(self):
        """Test that ensure_directories doesn't create when create_if_missing=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_root = Path(tmpdir)
            paths = Paths(mist_root=custom_root)
            
            # Don't create directories
            paths.ensure_directories(create_if_missing=False)
            
            # Verify directories don't exist (they weren't created)
            assert not paths.config.exists()
            assert not paths.data.exists()
    
    def test_ensure_directories_idempotent(self):
        """Test that ensure_directories can be called multiple times safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_root = Path(tmpdir)
            paths = Paths(mist_root=custom_root)
            
            # Call multiple times
            paths.ensure_directories(create_if_missing=True)
            paths.ensure_directories(create_if_missing=True)
            paths.ensure_directories(create_if_missing=True)
            
            # Should still work fine
            assert paths.config.exists()
            assert paths.data.exists()


class TestCrossPlatformCompatibility:
    """Test cross-platform path compatibility."""
    
    def test_paths_use_pathlib(self):
        """Test that all paths use pathlib.Path for cross-platform compatibility."""
        paths = Paths()
        
        # All properties should return Path objects
        assert isinstance(paths.config, Path)
        assert isinstance(paths.data, Path)
        assert isinstance(paths.src, Path)
        assert isinstance(paths.scripts, Path)
        assert isinstance(paths.migrations, Path)
        assert isinstance(paths.tests, Path)
        assert isinstance(paths.databases, Path)
        assert isinstance(paths.knowledge_graph, Path)
        assert isinstance(paths.vector_store, Path)
        assert isinstance(paths.feedback, Path)
        assert isinstance(paths.feedback_db, Path)
        assert isinstance(paths.embeddings, Path)
        assert isinstance(paths.embeddings_checkpoints, Path)
    
    def test_path_resolution(self):
        """Test that paths are resolved correctly."""
        paths = Paths()
        
        # All paths should be absolute (resolved)
        assert paths.config.is_absolute()
        assert paths.data.is_absolute()
        assert paths.mist_root.is_absolute()
    
    def test_path_separators(self):
        """Test that path separators are handled correctly."""
        paths = Paths()
        
        # Path objects handle separators automatically
        # Test that joining works correctly
        test_path = paths.data / "test" / "subdir" / "file.txt"
        assert isinstance(test_path, Path)
        # Should work on all platforms
        assert "test" in test_path.parts
        assert "subdir" in test_path.parts


class TestGlobalInstance:
    """Test global get_paths function."""
    
    def test_get_paths_returns_instance(self):
        """Test that get_paths returns a Paths instance."""
        paths = get_paths()
        assert isinstance(paths, Paths)
    
    def test_get_paths_singleton(self):
        """Test that get_paths returns the same instance."""
        paths1 = get_paths()
        paths2 = get_paths()
        assert paths1 is paths2
    
    def test_global_instance_paths_work(self):
        """Test that global instance paths work correctly."""
        paths = get_paths()
        assert isinstance(paths.config, Path)
        assert isinstance(paths.data, Path)


class TestRepr:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        paths = Paths()
        repr_str = repr(paths)
        assert "Paths" in repr_str
        assert "mist_root" in repr_str