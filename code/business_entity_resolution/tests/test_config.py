"""
Configuration and path resolution tests.
Verifies dynamic path resolution, DATASET_ROOT env-var override, and
path correctness when running from different working directories.
"""

import os
import tempfile
from pathlib import Path

import pytest

from business_entity_resolution.config import (
    BlockingConfig,
    ModelConfig,
    PathConfig,
    PipelineConfig,
    PROJECT_ROOT,
    find_dataset_root,
)


class TestProjectRoot:
    def test_project_root_is_path(self):
        assert isinstance(PROJECT_ROOT, Path)

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_project_root_not_src_dir(self):
        """PROJECT_ROOT should be the repo root, not the src dir."""
        assert PROJECT_ROOT.name != "src"
        assert PROJECT_ROOT.name != "business_entity_resolution"

    def test_project_root_contains_code_dir(self):
        """Repo root should contain the 'code' directory."""
        assert (PROJECT_ROOT / "code").exists()


class TestDatasetRootDiscovery:
    def test_env_var_override(self, tmp_path):
        """DATASET_ROOT env variable should override all other discovery."""
        # Create a fake dataset root with a train subdirectory
        fake_root = tmp_path / "my_dataset"
        (fake_root / "train").mkdir(parents=True)
        original = os.environ.get("DATASET_ROOT")
        try:
            os.environ["DATASET_ROOT"] = str(fake_root)
            result = find_dataset_root()
            assert result == fake_root.resolve()
        finally:
            if original is None:
                del os.environ["DATASET_ROOT"]
            else:
                os.environ["DATASET_ROOT"] = original

    def test_env_var_nonexistent_falls_through(self, tmp_path):
        """If DATASET_ROOT env var points to non-existent path, fall through to auto-discovery."""
        original = os.environ.get("DATASET_ROOT")
        try:
            os.environ["DATASET_ROOT"] = str(tmp_path / "does_not_exist")
            # Should not raise, just fall through
            result = find_dataset_root()
            assert isinstance(result, Path)
        finally:
            if original is None:
                del os.environ["DATASET_ROOT"]
            else:
                os.environ["DATASET_ROOT"] = original

    def test_default_returns_path(self):
        """find_dataset_root() should always return a Path, even if it doesn't exist."""
        result = find_dataset_root()
        assert isinstance(result, Path)


class TestPathConfig:
    def test_train_paths_under_dataset_root(self):
        config = PathConfig()
        assert config.train_source1.parent.name == "train"
        assert config.train_source2.parent.name == "train"
        assert config.train_source3.parent.name == "train"
        assert config.train_ground_truth.parent.name == "train"

    def test_test_paths_under_dataset_root(self):
        config = PathConfig()
        assert config.test_source1.parent.name == "test"
        assert config.test_source2.parent.name == "test"
        assert config.test_source3.parent.name == "test"

    def test_model_path_under_artifacts(self):
        config = PathConfig()
        assert config.model_path.parent == config.artifacts_dir

    def test_output_paths_under_output_dir(self):
        config = PathConfig()
        assert config.matching_results.parent == config.output_dir
        assert config.candidate_pairs.parent == config.output_dir

    def test_paths_are_absolute(self):
        config = PathConfig()
        assert config.train_source1.is_absolute()
        assert config.model_path.is_absolute()
        assert config.matching_results.is_absolute()

    def test_no_machine_specific_paths(self):
        """Paths must not contain any hardcoded machine-specific segments."""
        config = PathConfig()
        path_str = str(config.train_source1)
        assert "RAGHAV" not in path_str, "Hardcoded username found in path!"
        assert "New folder" not in path_str or str(PROJECT_ROOT) in path_str


class TestBlockingConfig:
    def test_defaults(self):
        bc = BlockingConfig()
        assert bc.max_candidates_per_entity == 80
        assert bc.max_block_size == 350
        assert bc.min_token_len == 3
        assert bc.name_prefix_len == 4

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OPTIRESOLVE_MAX_CANDIDATES", "60")
        bc = BlockingConfig()
        assert bc.max_candidates_per_entity == 60

    def test_explicit_override(self):
        bc = BlockingConfig(max_candidates_per_entity=100)
        assert bc.max_candidates_per_entity == 100


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig()
        assert mc.objective == "binary"
        assert mc.n_estimators == 450
        assert mc.learning_rate == 0.05


class TestPipelineConfig:
    def test_defaults(self):
        pc = PipelineConfig()
        assert pc.random_seed == 42
        assert pc.max_negatives_per_positive == 15
        assert pc.default_threshold == 0.910

    def test_dev_mode_override(self):
        """Simulate what run_pipeline.py does in --dev mode."""
        config = PipelineConfig()
        config.train_s1_limit = 50000
        config.val_s1_limit = 10000
        assert config.train_s1_limit == 50000
        assert config.val_s1_limit == 10000
