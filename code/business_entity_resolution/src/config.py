"""
Configuration module for Business Entity Resolution Pipeline.
Strictly zero external data; fully supports open-set country distribution (US, India, France).
All paths resolve dynamically relative to the repository project root.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Locate the root of the repository dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def find_dataset_root() -> Path:
    """Dynamically discover dataset directory relative to project root or environment."""
    if "DATASET_ROOT" in os.environ:
        p = Path(os.environ["DATASET_ROOT"]).resolve()
        if p.exists():
            return p

    candidates = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT.parent / "dataset",
        PROJECT_ROOT.parent / "dataset amazon" / "student_resource" / "dataset",
        Path(r"D:\dataset amazon\student_resource\dataset"),
    ]
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c.resolve()

    return (PROJECT_ROOT / "dataset").resolve()


@dataclass
class PathConfig:
    # Base dataset paths (auto-discovered)
    dataset_root: Path = field(default_factory=find_dataset_root)

    @property
    def train_source1(self) -> Path:
        return self.dataset_root / "train" / "train_source1.tsv"

    @property
    def train_source2(self) -> Path:
        return self.dataset_root / "train" / "train_source2.tsv"

    @property
    def train_source3(self) -> Path:
        return self.dataset_root / "train" / "train_source3.tsv"

    @property
    def train_ground_truth(self) -> Path:
        return self.dataset_root / "train" / "train_ground_truth.tsv"

    @property
    def test_source1(self) -> Path:
        return self.dataset_root / "test" / "test_source1.tsv"

    @property
    def test_source2(self) -> Path:
        return self.dataset_root / "test" / "test_source2.tsv"

    @property
    def test_source3(self) -> Path:
        return self.dataset_root / "test" / "test_source3.tsv"

    # Project-relative output paths
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")

    @property
    def matching_results(self) -> Path:
        return self.output_dir / "matching_results.tsv"

    @property
    def candidate_pairs(self) -> Path:
        return self.output_dir / "candidate_pairs.tsv"

    # Project-relative artifacts & model storage
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "lightgbm_er_model.joblib"


@dataclass
class BlockingConfig:
    max_candidates_per_entity: int = 80  # Optimized candidate safety cap
    min_token_len: int = 3
    name_prefix_len: int = 4
    max_block_size: int = 350
    sub_block_threshold: int = 350  # Apply secondary sub-blocking instead of hard deletion


@dataclass
class ModelConfig:
    objective: str = "binary"
    metric: str = "binary_logloss"
    boosting_type: str = "gbdt"
    n_estimators: int = 450
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 25
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    random_seed: int = 42

    # None = full scale execution
    train_s1_limit: Optional[int] = None
    val_s1_limit: Optional[int] = None
    max_negatives_per_positive: int = 15

    # Threshold tuning
    threshold_search_start: float = 0.65
    threshold_search_end: float = 0.96
    threshold_search_step: float = 0.02
    default_threshold: float = 0.910
