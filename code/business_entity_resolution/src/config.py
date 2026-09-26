"""
Configuration module for Business Entity Resolution Pipeline.
Strictly zero external data; fully supports open-set country distribution (US, India, France).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PathConfig:
    # Base dataset paths (defaults to official competition directory)
    dataset_root: Path = Path(r"D:\dataset amazon\student_resource\dataset")
    train_source1: Path = field(default_factory=lambda: Path(r"D:\dataset amazon\student_resource\dataset\train\train_source1.tsv"))
    train_source2: Path = field(default_factory=lambda: Path(r"D:\dataset amazon\student_resource\dataset\train\train_source2.tsv"))
    train_source3: Path = field(default_factory=lambda: Path(r"D:\dataset amazon\student_resource\dataset\train\train_source3.tsv"))
    train_ground_truth: Path = field(default_factory=lambda: Path(r"D:\dataset amazon\student_resource\dataset\train\train_ground_truth.tsv"))

    test_source1: Path = field(default_factory=lambda: Path(r"D:\dataset amazon\student_resource\dataset\test\test_source1.tsv"))
    test_source2: Path = field(default_factory=lambda: Path(r"D:\dataset amazon\student_resource\dataset\test\test_source2.tsv"))
    test_source3: Path = field(default_factory=lambda: Path(r"D:\dataset amazon\student_resource\dataset\test\test_source3.tsv"))

    # Output paths
    output_dir: Path = Path(r"d:\New folder (2)\output")
    matching_results: Path = Path(r"d:\New folder (2)\output\matching_results.tsv")
    candidate_pairs: Path = Path(r"d:\New folder (2)\output\candidate_pairs.tsv")

    # Artifacts & model storage
    artifacts_dir: Path = Path(r"d:\New folder (2)\artifacts")
    model_path: Path = Path(r"d:\New folder (2)\artifacts\lightgbm_er_model.joblib")


@dataclass
class BlockingConfig:
    max_candidates_per_entity: int = 40
    min_token_len: int = 3
    name_prefix_len: int = 4
    max_block_size: int = 400


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

    # Training limits (None = full scale, integer = subsample for fast dev/validation)
    train_s1_limit: Optional[int] = 80000
    val_s1_limit: Optional[int] = 15000
    max_negatives_per_positive: int = 15

    # Threshold tuning
    threshold_search_start: float = 0.65
    threshold_search_end: float = 0.96
    threshold_search_step: float = 0.02
    default_threshold: float = 0.88
