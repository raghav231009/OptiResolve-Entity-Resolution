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
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[dict] = None
        self.validation_loss_: Optional[float] = None

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
        stopping_rounds = getattr(self.config, "early_stopping_rounds", 30)
        callbacks = [lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=True)] if has_val else None

        self.clf.fit(
            X_train,
            y_train,
            eval_X=X_val if has_val else None,
            eval_y=y_val if has_val else None,
            eval_names=["val"] if has_val else None,
            callbacks=callbacks,
        )

        if has_val and hasattr(self.clf, "best_iteration_") and self.clf.best_iteration_ is not None:
            self.best_iteration_ = int(self.clf.best_iteration_)
            self.best_score_ = self.clf.best_score_ or {}
            val_scores = self.best_score_.get("val", {})
            val_loss = val_scores.get("binary_logloss", None)
            if val_loss is not None:
                self.validation_loss_ = float(val_loss)
            logger.info(
                f"Early stopping confirmed: Best iteration = {self.best_iteration_} / {self.config.n_estimators} "
                f"| Best validation binary logloss = {self.validation_loss_}"
            )
        else:
            self.best_iteration_ = self.config.n_estimators
            self.best_score_ = {}
            self.validation_loss_ = None

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
        if hasattr(self.clf, "best_iteration_") and self.clf.best_iteration_ is not None:
            self.best_iteration_ = int(self.clf.best_iteration_)
        if hasattr(self.clf, "best_score_") and self.clf.best_score_:
            self.best_score_ = self.clf.best_score_
            val_loss = self.best_score_.get("val", {}).get("binary_logloss")
            if val_loss is not None:
                self.validation_loss_ = float(val_loss)
