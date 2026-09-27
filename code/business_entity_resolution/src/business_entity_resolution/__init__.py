"""
Business Entity Resolution System Package.
"""

from .config import PipelineConfig, PathConfig, BlockingConfig, ModelConfig
from .normalization import strip_accents_and_normalize, clean_business_name, normalize_address
from .blocking import MultiIndexBlocker
from .features import compute_pair_features, FEATURE_NAMES
from .metrics import compute_macro_f05, compute_entity_f05
from .model import EntityResolutionModel
from .threshold import optimize_threshold, evaluate_threshold, ThresholdMetric
from .calibration import (
    compute_distribution_statistics,
    compute_ece,
    compute_brier_score,
    segment_probability_scores,
    analyze_threshold_sensitivity,
    evaluate_calibration_benefit,
)

__all__ = [
    "PipelineConfig",
    "PathConfig",
    "BlockingConfig",
    "ModelConfig",
    "strip_accents_and_normalize",
    "clean_business_name",
    "normalize_address",
    "MultiIndexBlocker",
    "compute_pair_features",
    "FEATURE_NAMES",
    "compute_macro_f05",
    "compute_entity_f05",
    "EntityResolutionModel",
    "optimize_threshold",
    "evaluate_threshold",
    "ThresholdMetric",
    "compute_distribution_statistics",
    "compute_ece",
    "compute_brier_score",
    "segment_probability_scores",
    "analyze_threshold_sensitivity",
    "evaluate_calibration_benefit",
]
