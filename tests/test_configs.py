"""
Tests for configuration file validation.
Verifies that all YAML config files are valid and contain required fields.
"""
import pytest
import yaml
from pathlib import Path
from typing import Dict, Any, List
from src.paths import Paths


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_required_keys(config: Dict[str, Any], required_keys: List[str], prefix: str = "") -> List[str]:
    """Validate that required keys exist in config dict."""
    missing = []
    for key in required_keys:
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in config:
            missing.append(full_key)
        elif isinstance(config[key], dict):
            # Recursively check nested dicts if needed
            pass
    return missing


class TestEmbeddingConfig:
    """Test embedding_config.yaml"""
    
    @pytest.fixture
    def config(self):
        """Load embedding config."""
        paths = Paths()
        config_path = paths.config / "embedding_config.yaml"
        return load_yaml_config(config_path)
    
    def test_file_loads(self, config):
        """Test that config file loads without errors."""
        assert config is not None
        assert isinstance(config, dict)
    
    def test_models_section(self, config):
        """Test models section structure."""
        assert "models" in config
        models = config["models"]
        
        # Fault code encoder
        assert "fault_code" in models
        fault_code = models["fault_code"]
        assert "model_name" in fault_code
        assert "projection_dim" in fault_code
        assert "max_length" in fault_code
        assert "device" in fault_code
        assert fault_code["projection_dim"] == 768
        
        # OBD data encoder
        assert "obd_data" in models
        obd_data = models["obd_data"]
        assert "input_dim" in obd_data
        assert "hidden_dim" in obd_data
        assert "output_dim" in obd_data
        assert "attention_heads" in obd_data
        assert obd_data["output_dim"] == 768
        
        # Fusion
        assert "fusion" in models
        fusion = models["fusion"]
        assert "type" in fusion
        assert "hidden_dim" in fusion
        assert "num_heads" in fusion
        assert "dropout" in fusion
        assert fusion["type"] == "cross_attention"
        
        # Output dimension
        assert "output_dimension" in models
        assert models["output_dimension"] == 768
    
    def test_training_section(self, config):
        """Test training section structure."""
        assert "training" in config
        training = config["training"]
        
        required_keys = ["batch_size", "learning_rate", "num_epochs", 
                        "warmup_steps", "weight_decay", "temperature"]
        missing = validate_required_keys(training, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        # Validate value types
        assert isinstance(training["batch_size"], int)
        assert isinstance(training["learning_rate"], (int, float))
        assert isinstance(training["num_epochs"], int)
        assert isinstance(training["temperature"], (int, float))
        assert 0 < training["temperature"] <= 1.0
    
    def test_fine_tuning_section(self, config):
        """Test fine_tuning section (if present)."""
        if "fine_tuning" in config:
            fine_tuning = config["fine_tuning"]
            assert "enabled" in fine_tuning
            assert isinstance(fine_tuning["enabled"], bool)


class TestRetrievalConfig:
    """Test retrieval_config.yaml"""
    
    @pytest.fixture
    def config(self):
        """Load retrieval config."""
        paths = Paths()
        config_path = paths.config / "retrieval_config.yaml"
        return load_yaml_config(config_path)
    
    def test_file_loads(self, config):
        """Test that config file loads without errors."""
        assert config is not None
        assert isinstance(config, dict)
    
    def test_vector_store_section(self, config):
        """Test vector_store section."""
        assert "vector_store" in config
        vs = config["vector_store"]
        
        required_keys = ["provider", "collection_name", "distance_metric", 
                        "vector_size", "url"]
        missing = validate_required_keys(vs, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        assert vs["provider"] == "qdrant"
        assert vs["distance_metric"] == "cosine"
        assert vs["vector_size"] == 768
    
    def test_retrieval_section(self, config):
        """Test retrieval section."""
        assert "retrieval" in config
        retrieval = config["retrieval"]
        
        required_keys = ["initial_k", "rerank_k", "final_k", "min_similarity"]
        missing = validate_required_keys(retrieval, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        # Validate K values are positive integers
        assert retrieval["initial_k"] > 0
        assert retrieval["rerank_k"] > 0
        assert retrieval["final_k"] > 0
        assert retrieval["rerank_k"] <= retrieval["initial_k"]
        assert retrieval["final_k"] <= retrieval["rerank_k"]
    
    def test_reranking_section(self, config):
        """Test reranking section."""
        assert "reranking" in config
        reranking = config["reranking"]
        
        required_keys = ["enabled", "provider", "model", "top_k"]
        missing = validate_required_keys(reranking, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        assert isinstance(reranking["enabled"], bool)
        assert reranking["provider"] in ["cohere", "local"]
    
    def test_ranking_section(self, config):
        """Test ranking weights section."""
        assert "ranking" in config
        ranking = config["ranking"]
        
        required_keys = ["embedding_similarity", "rerank_score", 
                        "kg_path_score", "feedback_score"]
        missing = validate_required_keys(ranking, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        # Validate weights sum to approximately 1.0
        total_weight = sum([
            ranking["embedding_similarity"],
            ranking["rerank_score"],
            ranking["kg_path_score"],
            ranking["feedback_score"]
        ])
        assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected ~1.0"
    
    def test_knowledge_graph_section(self, config):
        """Test knowledge_graph section."""
        assert "knowledge_graph" in config
        kg = config["knowledge_graph"]
        
        required_keys = ["enabled", "graph_path", "max_path_length", "min_path_score"]
        missing = validate_required_keys(kg, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        assert isinstance(kg["enabled"], bool)
        assert kg["max_path_length"] > 0
    
    def test_clarification_section(self, config):
        """Test clarification section."""
        assert "clarification" in config
        clarification = config["clarification"]
        
        required_keys = ["enabled", "ambiguity_threshold", "score_variance_threshold",
                        "max_questions", "max_clarifications_per_session"]
        missing = validate_required_keys(clarification, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        assert isinstance(clarification["enabled"], bool)
        assert 0 <= clarification["ambiguity_threshold"] <= 1.0
        assert clarification["max_questions"] > 0


class TestLLMConfig:
    """Test llm_config.yaml"""
    
    @pytest.fixture
    def config(self):
        """Load LLM config."""
        paths = Paths()
        config_path = paths.config / "llm_config.yaml"
        return load_yaml_config(config_path)
    
    def test_file_loads(self, config):
        """Test that config file loads without errors."""
        assert config is not None
        assert isinstance(config, dict)
    
    def test_providers_section(self, config):
        """Test providers section."""
        assert "providers" in config
        providers = config["providers"]
        
        assert "primary" in providers
        assert "fallback" in providers
        assert isinstance(providers["fallback"], list)
        assert len(providers["fallback"]) > 0
    
    def test_openai_section(self, config):
        """Test OpenAI provider configuration."""
        assert "openai" in config
        openai = config["openai"]
        
        required_keys = ["model", "api_key_env", "temperature", "max_tokens", "timeout"]
        missing = validate_required_keys(openai, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        assert isinstance(openai["temperature"], (int, float))
        assert 0 <= openai["temperature"] <= 2.0
        assert openai["max_tokens"] > 0
    
    def test_anthropic_section(self, config):
        """Test Anthropic provider configuration."""
        assert "anthropic" in config
        anthropic = config["anthropic"]
        
        required_keys = ["model", "api_key_env", "temperature", "max_tokens", "timeout"]
        missing = validate_required_keys(anthropic, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        assert isinstance(anthropic["temperature"], (int, float))
        assert 0 <= anthropic["temperature"] <= 1.0
    
    def test_open_source_section(self, config):
        """Test open-source provider configuration."""
        assert "open_source" in config
        open_source = config["open_source"]
        
        required_keys = ["provider", "model", "base_url", "temperature", "max_tokens"]
        missing = validate_required_keys(open_source, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        assert open_source["provider"] == "ollama"
    
    def test_prompts_section(self, config):
        """Test prompt templates section."""
        assert "prompts" in config
        prompts = config["prompts"]
        
        assert "clarification" in prompts
        clarification = prompts["clarification"]
        assert "system" in clarification
        assert "user_template" in clarification
        assert "{fault_codes}" in clarification["user_template"]
        
        assert "query_expansion" in prompts
        query_expansion = prompts["query_expansion"]
        assert "system" in query_expansion
        assert "user_template" in query_expansion
        assert "{original_query}" in query_expansion["user_template"]


class TestTrainingConfig:
    """Test training_config.yaml"""
    
    @pytest.fixture
    def config(self):
        """Load training config."""
        paths = Paths()
        config_path = paths.config / "training_config.yaml"
        return load_yaml_config(config_path)
    
    def test_file_loads(self, config):
        """Test that config file loads without errors."""
        assert config is not None
        assert isinstance(config, dict)
    
    def test_training_section(self, config):
        """Test training section."""
        assert "training" in config
        training = config["training"]
        
        required_keys = ["batch_size", "learning_rate", "num_epochs", 
                        "warmup_steps", "weight_decay", "temperature"]
        missing = validate_required_keys(training, required_keys)
        assert len(missing) == 0, f"Missing keys: {missing}"
        
        assert isinstance(training["batch_size"], int)
        assert training["batch_size"] > 0
        assert isinstance(training["learning_rate"], (int, float))
        assert training["learning_rate"] > 0
    
    def test_fine_tuning_section(self, config):
        """Test fine_tuning section."""
        assert "fine_tuning" in config
        fine_tuning = config["fine_tuning"]
        
        assert "enabled" in fine_tuning
        assert isinstance(fine_tuning["enabled"], bool)
        assert "checkpoint_interval" in fine_tuning
        assert "min_feedback_samples" in fine_tuning
        assert "validation_split" in fine_tuning
        
        assert 0 <= fine_tuning["validation_split"] <= 1.0
    
    def test_reward_model_section(self, config):
        """Test reward_model section."""
        assert "reward_model" in config
        reward_model = config["reward_model"]
        
        assert "input_dim" in reward_model
        assert "hidden_dim" in reward_model
        assert "learning_rate" in reward_model
        
        assert reward_model["input_dim"] == 768
    
    def test_active_learning_section(self, config):
        """Test active_learning section."""
        assert "active_learning" in config
        active_learning = config["active_learning"]
        
        assert "enabled" in active_learning
        assert isinstance(active_learning["enabled"], bool)
        assert "uncertainty_threshold" in active_learning
        assert "score_variance_threshold" in active_learning
        assert "top_n_for_analysis" in active_learning
        assert "batch_size" in active_learning
        assert "sampling_strategy" in active_learning
        
        assert active_learning["sampling_strategy"] in ["uncertainty", "random", "diversity"]
        assert isinstance(active_learning["uncertainty_threshold"], (int, float))
        assert isinstance(active_learning["score_variance_threshold"], (int, float))
        assert isinstance(active_learning["top_n_for_analysis"], int)
        assert active_learning["top_n_for_analysis"] > 0


class TestConfigFilesExist:
    """Test that all required config files exist."""
    
    def test_all_config_files_exist(self):
        """Verify all 4 config files exist."""
        paths = Paths()
        config_dir = paths.config
        
        required_files = [
            "embedding_config.yaml",
            "retrieval_config.yaml",
            "llm_config.yaml",
            "training_config.yaml"
        ]
        
        for filename in required_files:
            config_path = config_dir / filename
            assert config_path.exists(), f"Config file {filename} does not exist at {config_path}"
            assert config_path.is_file(), f"{filename} is not a file"


class TestConfigYAMLSyntax:
    """Test that all config files have valid YAML syntax."""
    
    @pytest.mark.parametrize("config_file", [
        "embedding_config.yaml",
        "retrieval_config.yaml",
        "llm_config.yaml",
        "training_config.yaml"
    ])
    def test_yaml_syntax_valid(self, config_file):
        """Test that each config file has valid YAML syntax."""
        paths = Paths()
        config_path = paths.config / config_file
        
        # Should not raise exception
        config = load_yaml_config(config_path)
        assert config is not None
        assert isinstance(config, dict)
