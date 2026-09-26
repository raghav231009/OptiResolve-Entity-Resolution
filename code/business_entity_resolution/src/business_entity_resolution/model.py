"""
Model Training Module for Business Entity Resolution.
Trains LightGBM GBDT classifier with hard-negative mining and early stopping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np

from .config import ModelConfig
from .features import FEATURE_NAMES


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
        """Train LightGBM binary classifier."""
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

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None

        self.clf.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

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
