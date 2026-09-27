"""
Model Training Module for Business Entity Resolution.
Trains LightGBM GBDT classifier with hard-negative mining, active validation monitoring,
and early stopping.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ModelConfig
from .features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def _serialize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types and dicts for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class EntityResolutionModel:
    """Wrapper around LightGBM Classifier for pairwise entity matching."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.clf: Optional[lgb.LGBMClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[dict] = None
        self.validation_loss_: Optional[float] = None

    @property
    def n_features_in_(self) -> int:
        if self.clf is not None and hasattr(self.clf, "n_features_in_"):
            return int(self.clf.n_features_in_)
        return len(FEATURE_NAMES)

    @property
    def feature_name_(self) -> List[str]:
        if self.clf is not None and hasattr(self.clf, "feature_name_") and self.clf.feature_name_:
            return list(self.clf.feature_name_)
        return list(FEATURE_NAMES)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Train LightGBM binary classifier with verified early stopping."""
        self.clf = lgb.LGBMClassifier(
            objective=self.config.objective,
            metric=self.config.metric,
            boosting_type=self.config.boosting_type,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=-1,
        )

        has_val = (
            X_val is not None
            and y_val is not None
            and len(X_val) > 0
            and len(y_val) > 0
        )

        # Requirement: Verify LightGBM receives exactly 23 columns matching FEATURE_NAMES
        if X_train is not None and len(X_train) > 0:
            assert X_train.shape[1] == len(FEATURE_NAMES) == 23, (
                f"Feature dimension mismatch: X_train has {X_train.shape[1]} columns, "
                f"expected exactly {len(FEATURE_NAMES)} (23)"
            )
        if has_val:
            assert X_val.shape[1] == len(FEATURE_NAMES) == 23, (
                f"Feature dimension mismatch: X_val has {X_val.shape[1]} columns, "
                f"expected exactly {len(FEATURE_NAMES)} (23)"
            )

        stopping_rounds = getattr(self.config, "early_stopping_rounds", 30)
        callbacks = [lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=True)] if has_val else None

        # LightGBM 4.7.0 pinned API: eval_X and eval_y are passed to avoid LGBMDeprecationWarning
        self.clf.fit(
            X_train,
            y_train,
            feature_name=FEATURE_NAMES,
            eval_X=X_val if has_val else None,
            eval_y=y_val if has_val else None,
            eval_names=["val"] if has_val else None,
            callbacks=callbacks,
        )

        if has_val and hasattr(self.clf, "best_iteration_") and self.clf.best_iteration_ is not None and self.clf.best_iteration_ > 0:
            self.best_iteration_ = int(self.clf.best_iteration_)
            self.best_score_ = dict(self.clf.best_score_) if self.clf.best_score_ else {}
            val_scores = self.best_score_.get("val", {})
            val_loss = val_scores.get(self.config.metric, None) or val_scores.get("binary_logloss", None)
            if val_loss is not None:
                self.validation_loss_ = float(val_loss)
            logger.info(
                f"Early stopping confirmed: Best iteration = {self.best_iteration_} / {self.config.n_estimators} "
                f"| Best validation {self.config.metric} = {self.validation_loss_}"
            )
        else:
            self.best_iteration_ = int(self.config.n_estimators)
            self.best_score_ = {}
            self.validation_loss_ = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive match."""
        if self.clf is None:
            raise ValueError("Model has not been trained yet.")
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        assert X.shape[1] == len(FEATURE_NAMES) == 23, (
            f"Feature dimension mismatch: input X has {X.shape[1]} columns, "
            f"expected exactly {len(FEATURE_NAMES)} (23)"
        )
        num_iter = self.best_iteration_ if (self.best_iteration_ is not None and self.best_iteration_ > 0) else None

        if hasattr(self.clf, "feature_name_") and self.clf.feature_name_:
            cols = list(self.clf.feature_name_)
            if not isinstance(X, pd.DataFrame):
                X_input = pd.DataFrame(X, columns=cols)
            else:
                X_input = X
        else:
            X_input = X

        probs = self.clf.predict_proba(X_input, num_iteration=num_iter)[:, 1]
        return np.asarray(probs, dtype=np.float32)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return mapping of feature name to gain importance."""
        if self.clf is None:
            return {}
        importances = self.clf.feature_importances_
        return dict(sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True))

    def save(self, filepath: Path):
        """Persist model artifact and companion metadata JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, filepath)

        metadata_path = filepath.with_suffix(".json")
        metadata = {
            "model_type": "LightGBM",
            "lightgbm_version": lgb.__version__,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "best_iteration": self.best_iteration_,
            "best_score": _serialize_for_json(self.best_score_ or {}),
            "validation_loss": self.validation_loss_,
            "hyperparameters": {
                "objective": self.config.objective,
                "metric": self.config.metric,
                "boosting_type": self.config.boosting_type,
                "n_estimators": self.config.n_estimators,
                "learning_rate": self.config.learning_rate,
                "num_leaves": self.config.num_leaves,
                "max_depth": self.config.max_depth,
                "min_child_samples": self.config.min_child_samples,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bytree,
                "reg_alpha": self.config.reg_alpha,
                "reg_lambda": self.config.reg_lambda,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
                "early_stopping_rounds": getattr(self.config, "early_stopping_rounds", 30),
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model serialized to {filepath} (companion metadata: {metadata_path})")

    def load(self, filepath: Path):
        """Load persisted model artifact and companion metadata."""
        filepath = Path(filepath)
        self.clf = joblib.load(filepath)
        if hasattr(self.clf, "best_iteration_") and self.clf.best_iteration_ is not None and self.clf.best_iteration_ > 0:
            self.best_iteration_ = int(self.clf.best_iteration_)
        if hasattr(self.clf, "best_score_") and self.clf.best_score_:
            self.best_score_ = dict(self.clf.best_score_)
            val_loss = self.best_score_.get("val", {}).get(self.config.metric) or self.best_score_.get("val", {}).get("binary_logloss")
            if val_loss is not None:
                self.validation_loss_ = float(val_loss)

        metadata_path = filepath.with_suffix(".json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("best_iteration") is not None:
                    self.best_iteration_ = int(meta["best_iteration"])
                if meta.get("validation_loss") is not None:
                    self.validation_loss_ = float(meta["validation_loss"])
                if meta.get("best_score"):
                    self.best_score_ = meta["best_score"]
            except Exception as e:
                logger.warning(f"Could not load companion metadata JSON from {metadata_path}: {e}")
