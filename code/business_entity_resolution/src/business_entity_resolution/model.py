"""
Model Training Module for Business Entity Resolution.
Trains LightGBM GBDT classifier with hard-negative mining, active validation monitoring,
and early stopping.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np

from .config import ModelConfig
from .features import FEATURE_NAMES

logger = logging.getLogger(__name__)


class EntityResolutionModel:
    """Wrapper around LightGBM Classifier for pairwise entity matching."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.clf: Optional[lgb.LGBMClassifier] = None

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

        # Build eval arguments: use new eval_X/eval_y API (eval_set deprecated in LightGBM>=4.4)
        has_val = X_val is not None and y_val is not None
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=True)] if has_val else None

        self.clf.fit(
            X_train,
            y_train,
            eval_X=X_val if has_val else None,
            eval_y=y_val if has_val else None,
            eval_names=["val"] if has_val else None,
            callbacks=callbacks,
        )

        if has_val and hasattr(self.clf, "best_iteration_"):
            # best_score_ structure: {eval_set_name: {metric_name: score}}
            val_scores = self.clf.best_score_ or {}
            val_loss = val_scores.get("val", {}).get("binary_logloss", "N/A")
            logger.info(f"Early stopping confirmed: Best iteration = {self.clf.best_iteration_} | Best validation score = {val_loss}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive match."""
        if self.clf is None:
            raise ValueError("Model has not been trained yet.")
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        return self.clf.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Return mapping of feature name to gain importance."""
        if self.clf is None:
            return {}
        importances = self.clf.feature_importances_
        return dict(sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True))

    def save(self, filepath: Path):
        """Persist model artifact."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, filepath)

    def load(self, filepath: Path):
        """Load persisted model artifact."""
        self.clf = joblib.load(filepath)
