"""
Model Tests.
Verifies training, saving, loading, and prediction of EntityResolutionModel.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from business_entity_resolution.model import EntityResolutionModel


def make_simple_data(n_pos=30, n_neg=60, n_features=23, seed=42):
    rng = np.random.default_rng(seed)
    X_pos = rng.uniform(0.7, 1.0, size=(n_pos, n_features)).astype(np.float32)
    X_neg = rng.uniform(0.0, 0.4, size=(n_neg, n_features)).astype(np.float32)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)
    return X, y


class TestModelTraining:
    def test_train_without_validation(self):
        X, y = make_simple_data()
        model = EntityResolutionModel()
        model.train(X, y)
        assert model.clf is not None

    def test_train_with_validation(self):
        X_train, y_train = make_simple_data(n_pos=20, n_neg=40)
        X_val, y_val = make_simple_data(n_pos=10, n_neg=20, seed=99)
        model = EntityResolutionModel()
        model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        assert model.clf is not None

    def test_predict_proba_shape(self):
        X, y = make_simple_data()
        model = EntityResolutionModel()
        model.train(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X),)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_proba_empty_input(self):
        X, y = make_simple_data()
        model = EntityResolutionModel()
        model.train(X, y)
        empty = np.empty((0, 23), dtype=np.float32)
        probs = model.predict_proba(empty)
        assert len(probs) == 0

    def test_predict_before_train_raises(self):
        model = EntityResolutionModel()
        X = np.zeros((5, 23), dtype=np.float32)
        with pytest.raises(ValueError, match="not been trained"):
            model.predict_proba(X)

    def test_feature_importances_non_empty(self):
        X, y = make_simple_data()
        model = EntityResolutionModel()
        model.train(X, y)
        importances = model.get_feature_importances()
        assert len(importances) == 23
        assert all(v >= 0 for v in importances.values())

    def test_feature_importances_before_train_empty(self):
        model = EntityResolutionModel()
        assert model.get_feature_importances() == {}


class TestModelPersistence:
    def test_save_and_load(self):
        X, y = make_simple_data()
        model = EntityResolutionModel()
        model.train(X, y)
        original_probs = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.joblib"
            model.save(path)
            assert path.exists()

            loaded_model = EntityResolutionModel()
            loaded_model.load(path)
            loaded_probs = loaded_model.predict_proba(X)

        np.testing.assert_allclose(original_probs, loaded_probs, rtol=1e-5)

    def test_save_creates_parent_dirs(self):
        X, y = make_simple_data()
        model = EntityResolutionModel()
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "model.joblib"
            model.save(path)
            assert path.exists()
