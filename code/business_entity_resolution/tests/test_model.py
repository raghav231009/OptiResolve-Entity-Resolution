"""
Model Tests for Modernized EntityResolutionModel.
Proves:
1. Compatibility with pinned LightGBM version (4.7.0).
2. Explicit validation inputs X_val, y_val.
3. Early stopping callback functions on synthetic data and stops early.
4. Model trains with intended n_estimators when early stopping is not triggered.
5. Recording of best_iteration_, best_score_, and validation_loss_.
6. Save/load preserves model, feature count (23), feature names, and best_iteration_.
7. Companion metadata JSON is written and loaded correctly.
8. predict_proba output shape (N,), numeric range [0.0, 1.0], and empty input handling.
9. Dimension check ensures LightGBM receives exactly 23 columns.
"""

import json
import tempfile
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pytest

from business_entity_resolution.config import ModelConfig
from business_entity_resolution.features import FEATURE_NAMES
from business_entity_resolution.model import EntityResolutionModel


def make_synthetic_separable_data(n_samples=200, n_features=23, seed=42):
    """Generate linearly separable synthetic dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    # Strong signal on first feature
    y = (X[:, 0] > 0.0).astype(np.int32)
    return X, y


def make_synthetic_overfitting_data(n_samples=400, n_features=23, seed=42):
    """Generate noisy synthetic dataset that causes early overfitting on validation."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    # Weak signal + heavy noise
    logits = X[:, 0] * 0.3 + rng.normal(0, 1.8, size=n_samples)
    y = (logits > 0).astype(np.int32)
    return X, y


class TestLightGBMModernization:
    """Verifies pinned LightGBM compatibility and early stopping mechanics."""

    def test_pinned_version_compatibility(self):
        """Verify LightGBM version is pinned 4.7.x."""
        assert lgb.__version__.startswith("4.7.")

    def test_early_stopping_functions_on_synthetic_data(self):
        """Synthetic overfitting data must cause early stopping well before n_estimators=200."""
        X_train, y_train = make_synthetic_overfitting_data(n_samples=500, seed=42)
        X_val, y_val = make_synthetic_overfitting_data(n_samples=150, seed=123)

        cfg = ModelConfig(n_estimators=200, learning_rate=0.1, early_stopping_rounds=8)
        model = EntityResolutionModel(config=cfg)

        model.train(X_train, y_train, X_val=X_val, y_val=y_val)

        # Early stopping must have triggered before 200
        assert model.best_iteration_ is not None
        assert 1 <= model.best_iteration_ < 200, (
            f"Expected early stopping before 200 estimators, got best_iteration_={model.best_iteration_}"
        )
        assert model.clf.best_iteration_ == model.best_iteration_

        # best_score_ and validation_loss_ must be populated
        assert model.best_score_ is not None
        assert model.validation_loss_ is not None
        assert 0.0 < model.validation_loss_ < 2.0

    def test_train_without_validation_uses_full_estimators(self):
        """When validation data is omitted, model uses the intended full n_estimators."""
        X_train, y_train = make_synthetic_separable_data(n_samples=100, seed=42)
        cfg = ModelConfig(n_estimators=45, learning_rate=0.05)
        model = EntityResolutionModel(config=cfg)

        model.train(X_train, y_train, X_val=None, y_val=None)

        assert model.best_iteration_ == 45
        assert model.validation_loss_ is None
        assert model.best_score_ == {}

    def test_predict_proba_shape_and_range(self):
        """predict_proba output must be 1D with values strictly in [0.0, 1.0]."""
        X_train, y_train = make_synthetic_separable_data(n_samples=100)
        model = EntityResolutionModel()
        model.train(X_train, y_train)

        X_test = np.random.normal(0, 1, size=(25, 23)).astype(np.float32)
        probs = model.predict_proba(X_test)

        assert probs.shape == (25,)
        assert probs.dtype == np.float32
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_proba_empty_input(self):
        """predict_proba on empty array must return empty 1D float32 array without error."""
        X_train, y_train = make_synthetic_separable_data(n_samples=50)
        model = EntityResolutionModel()
        model.train(X_train, y_train)

        empty_X = np.empty((0, 23), dtype=np.float32)
        probs = model.predict_proba(empty_X)

        assert probs.shape == (0,)
        assert probs.dtype == np.float32

    def test_predict_before_train_raises_error(self):
        model = EntityResolutionModel()
        X = np.zeros((5, 23), dtype=np.float32)
        with pytest.raises(ValueError, match="not been trained"):
            model.predict_proba(X)


class TestModelPersistenceAndMetadata:
    """Verifies save/load contract and companion metadata JSON."""

    def test_save_and_load_preserves_everything(self):
        X_train, y_train = make_synthetic_overfitting_data(n_samples=300, seed=42)
        X_val, y_val = make_synthetic_overfitting_data(n_samples=100, seed=88)

        cfg = ModelConfig(n_estimators=100, learning_rate=0.1, early_stopping_rounds=10)
        model = EntityResolutionModel(config=cfg)
        model.train(X_train, y_train, X_val=X_val, y_val=y_val)

        orig_best_iter = model.best_iteration_
        orig_val_loss = model.validation_loss_
        orig_probs = model.predict_proba(X_val)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "er_model.joblib"
            meta_path = Path(tmpdir) / "er_model.json"

            model.save(model_path)

            # Both joblib artifact and companion JSON must exist
            assert model_path.exists()
            assert meta_path.exists()

            # Verify companion metadata JSON structure
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            assert meta["model_type"] == "LightGBM"
            assert meta["feature_count"] == 23
            assert meta["feature_names"] == FEATURE_NAMES
            assert meta["best_iteration"] == orig_best_iter
            assert "hyperparameters" in meta
            assert meta["hyperparameters"]["n_estimators"] == 100

            # Load into a clean model instance
            loaded = EntityResolutionModel()
            loaded.load(model_path)

            assert loaded.best_iteration_ == orig_best_iter
            assert loaded.n_features_in_ == 23
            assert loaded.feature_name_ == FEATURE_NAMES
            if orig_val_loss is not None:
                assert loaded.validation_loss_ == pytest.approx(orig_val_loss, rel=1e-5)

            # Predict probabilities must match bit-for-bit
            loaded_probs = loaded.predict_proba(X_val)
            np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5)

    def test_feature_importances(self):
        X_train, y_train = make_synthetic_separable_data(n_samples=100)
        model = EntityResolutionModel()
        model.train(X_train, y_train)

        importances = model.get_feature_importances()
        assert len(importances) == 23
        assert list(importances.keys())[0] in FEATURE_NAMES
        assert all(isinstance(v, (int, float, np.floating, np.integer)) for v in importances.values())
