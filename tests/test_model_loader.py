"""Unit tests for model loader."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from app.core import model_loader


class TestModelLoader:

    def setup_method(self):
        """Reset model loader state before each test."""
        model_loader._ml_models = {}
        model_loader._tfidf = None
        model_loader._dl_model = None
        model_loader._dl_tokenizer = None

    def test_available_ml_models_empty_initially(self):
        assert model_loader.available_ml_models() == []

    def test_is_dl_available_false_initially(self):
        assert model_loader.is_dl_available() is False

    def test_get_tfidf_none_when_not_loaded(self):
        assert model_loader.get_tfidf() is None

    def test_get_ml_model_none_for_unknown(self):
        assert model_loader.get_ml_model("nonexistent_model") is None

    def test_load_ml_models_missing_files(self, tmp_path):
        """Loading from non-existent paths logs warning but doesn't crash."""
        registry = {
            "logistic_regression": tmp_path / "lr.joblib",
            "svm": tmp_path / "svm.joblib",
        }
        tfidf_path = tmp_path / "tfidf.joblib"

        # Should not raise, just log warnings
        model_loader.load_ml_models(registry, tfidf_path)
        assert model_loader.available_ml_models() == []
        assert model_loader.get_tfidf() is None

    def test_load_ml_models_with_mock(self, tmp_path):
        """Loading actual joblib files works correctly."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Create and save a tiny real model
        tfidf = TfidfVectorizer(max_features=10)
        X = tfidf.fit_transform(["fake review", "real review", "great product", "terrible product"])
        y = [1, 0, 0, 1]
        lr = LogisticRegression().fit(X, y)

        tfidf_path = tmp_path / "tfidf.joblib"
        lr_path = tmp_path / "lr.joblib"
        joblib.dump(tfidf, tfidf_path)
        joblib.dump(lr, lr_path)

        registry = {"logistic_regression": lr_path}
        model_loader.load_ml_models(registry, tfidf_path)

        assert model_loader.get_tfidf() is not None
        assert model_loader.get_ml_model("logistic_regression") is not None
        assert "logistic_regression" in model_loader.available_ml_models()

    def test_load_dl_model_empty_dir(self, tmp_path):
        """Loading DL model from empty dir doesn't crash."""
        empty_dir = tmp_path / "distilbert"
        empty_dir.mkdir()
        model_loader.load_dl_model(empty_dir)
        assert model_loader.is_dl_available() is False

    def test_load_dl_model_missing_dir(self, tmp_path):
        """Loading DL model from missing dir doesn't crash."""
        missing_dir = tmp_path / "nonexistent"
        model_loader.load_dl_model(missing_dir)
        assert model_loader.is_dl_available() is False
