"""
Configuration module for Business Entity Resolution Pipeline.
Strictly zero external data; fully supports open-set country distribution (US, India, France).
All paths resolve dynamically relative to the repository project root.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Locate the root of the repository dynamically.
# config.py is at: <repo>/code/business_entity_resolution/src/business_entity_resolution/config.py
# parents: [0]=business_entity_resolution, [1]=src, [2]=business_entity_resolution, [3]=code, [4]=<repo>
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def find_dataset_root() -> Path:
    """Dynamically discover dataset directory relative to project root or environment.

    Resolution order:
      1. DATASET_ROOT environment variable (if set and exists)
      2. <PROJECT_ROOT>/data
      3. <PROJECT_ROOT>/dataset

    If none found, returns <PROJECT_ROOT>/dataset as default (will raise
    a clear FileNotFoundError at data-load time if it doesn't exist).
    """
    if "DATASET_ROOT" in os.environ:
        p = Path(os.environ["DATASET_ROOT"]).resolve()
        if p.exists():
            return p

    candidates = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "dataset",
    ]
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c.resolve()

    # Return a sensible default; pipeline will fail with a clear path in the error
    return (PROJECT_ROOT / "dataset").resolve()


def find_output_dir() -> Path:
    """Dynamically discover output directory relative to project root or environment."""
    if "OPTIRESOLVE_OUTPUT_DIR" in os.environ:
        return Path(os.environ["OPTIRESOLVE_OUTPUT_DIR"]).resolve()
    return (PROJECT_ROOT / "output").resolve()


def find_artifacts_dir() -> Path:
    """Dynamically discover artifacts directory relative to project root or environment."""
    if "OPTIRESOLVE_ARTIFACTS_DIR" in os.environ:
        return Path(os.environ["OPTIRESOLVE_ARTIFACTS_DIR"]).resolve()
    return (PROJECT_ROOT / "artifacts").resolve()


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
    output_dir: Path = field(default_factory=find_output_dir)

    @property
    def matching_results(self) -> Path:
        return self.output_dir / "matching_results.tsv"

    @property
    def candidate_pairs(self) -> Path:
        return self.output_dir / "candidate_pairs.tsv"

    # Project-relative artifacts & model storage
    artifacts_dir: Path = field(default_factory=find_artifacts_dir)

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "lightgbm_er_model.joblib"

    def validate_train_dataset_exists(self) -> None:
        """Verify required training datasets exist or raise descriptive FileNotFoundError."""
        required = [
            ("train_source1", self.train_source1),
            ("train_source2", self.train_source2),
            ("train_source3", self.train_source3),
            ("train_ground_truth", self.train_ground_truth),
        ]
        for name, p in required:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required training dataset file '{name}' not found: {p}\n"
                    f"Resolved DATASET_ROOT is: {self.dataset_root}\n"
                    f"Please verify that the dataset exists or set the DATASET_ROOT environment variable."
                )

    def validate_test_dataset_exists(self) -> None:
        """Verify required test datasets exist or raise descriptive FileNotFoundError."""
        required = [
            ("test_source1", self.test_source1),
            ("test_source2", self.test_source2),
            ("test_source3", self.test_source3),
        ]
        for name, p in required:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required test dataset file '{name}' not found: {p}\n"
                    f"Resolved DATASET_ROOT is: {self.dataset_root}\n"
                    f"Please verify that the dataset exists or set the DATASET_ROOT environment variable."
                )


def _default_max_candidates() -> int:
    val = os.environ.get("OPTIRESOLVE_MAX_CANDIDATES")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return 80


@dataclass
class BlockingConfig:
    max_candidates_per_entity: int = field(default_factory=_default_max_candidates)
    min_token_len: int = 3
    name_prefix_len: int = 4
    max_block_size: int = 350
    sub_block_threshold: int = 350  # Apply secondary sub-blocking instead of hard deletion
    capping_strategy: str = "tiered"  # Options: "tiered", "current", "arbitrary"


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
    early_stopping_rounds: int = 30
    final_n_estimators: Optional[int] = None


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    random_seed: int = 42

    # Training execution mode: "production", "final-train", "experiment", "development", or "dev-train"
    training_mode: str = "production"

    # None = full scale execution (100% of available dataset used in production)
    train_s1_limit: Optional[int] = None
    val_s1_limit: Optional[int] = None
    max_negatives_per_positive: int = 15
    min_negatives_per_zero_positive_entity: int = 10

    # Threshold tuning
    threshold_search_start: float = 0.50
    threshold_search_end: float = 0.99
    threshold_search_step: float = 0.01
    threshold_fine_step: float = 0.002
    threshold_fine_window: float = 0.03
    default_threshold: float = 0.910

    allow_missing_targets: bool = False

    def __post_init__(self):
        if self.is_production and self.allow_missing_targets:
            raise ValueError(
                "Production training violation: allow_missing_targets cannot be True in production mode! "
                "Production requires 100% of ground-truth target entities to prevent incomplete positive labels."
            )

    @property
    def is_production(self) -> bool:
        return self.training_mode in ("production", "final-train")

    @property
    def is_final_train(self) -> bool:
        return self.training_mode in ("production", "final-train")

    @property
    def is_development(self) -> bool:
        return self.training_mode in ("development", "dev-train")

    @property
    def is_dev_train(self) -> bool:
        return self.training_mode in ("development", "dev-train", "experiment")

    @property
    def is_experiment(self) -> bool:
        return self.training_mode == "experiment"
