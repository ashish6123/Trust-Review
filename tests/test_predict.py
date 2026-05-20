"""Unit tests for prediction logic."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.ml.predict import predict_single, predict_batch, LABEL_MAP


class TestLabelMap:

    def test_label_map_keys(self):
        assert 0 in LABEL_MAP
        assert 1 in LABEL_MAP

    def test_label_map_values(self):
        assert LABEL_MAP[0] == "Real"
        assert LABEL_MAP[1] == "Fake"


class TestPredictSingle:

    def _make_mock_lr(self):
        """Create a mock Logistic Regression model."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_model.predict.return_value = np.array([1])
        return mock_model

    def _make_mock_tfidf(self):
        mock_tfidf = MagicMock()
        mock_tfidf.transform.return_value = MagicMock()
        return mock_tfidf

    def test_raises_when_model_not_loaded(self):
        with patch("app.core.model_loader.get_tfidf", return_value=None), \
             patch("app.core.model_loader.get_ml_model", return_value=None):
            with pytest.raises(RuntimeError):
                predict_single("Some review text", model_type="ml")

    def test_returns_dict_with_required_keys(self):
        mock_model = self._make_mock_lr()
        mock_tfidf = self._make_mock_tfidf()

        with patch("app.core.model_loader.get_tfidf", return_value=mock_tfidf), \
             patch("app.core.model_loader.get_ml_model", return_value=mock_model):
            result = predict_single("This product is great", model_type="ml")

        assert isinstance(result, dict)
        assert "text" in result
        assert "label" in result
        assert "confidence" in result
        assert "model_used" in result

    def test_label_is_valid(self):
        mock_model = self._make_mock_lr()
        mock_tfidf = self._make_mock_tfidf()

        with patch("app.core.model_loader.get_tfidf", return_value=mock_tfidf), \
             patch("app.core.model_loader.get_ml_model", return_value=mock_model):
            result = predict_single("Some review text", model_type="ml")

        assert result["label"] in ("Real", "Fake")

    def test_confidence_between_0_and_1(self):
        mock_model = self._make_mock_lr()
        mock_tfidf = self._make_mock_tfidf()

        with patch("app.core.model_loader.get_tfidf", return_value=mock_tfidf), \
             patch("app.core.model_loader.get_ml_model", return_value=mock_model):
            result = predict_single("Some review text", model_type="ml")

        assert 0.0 <= result["confidence"] <= 1.0

    def test_svm_uses_decision_function(self):
        """SVM (no predict_proba) falls back to sigmoid on decision_function."""
        mock_svm = MagicMock(spec=[])  # no predict_proba attribute
        mock_svm.decision_function = MagicMock(return_value=np.array([2.5]))
        mock_svm.predict = MagicMock(return_value=np.array([1]))
        mock_tfidf = self._make_mock_tfidf()

        with patch("app.core.model_loader.get_tfidf", return_value=mock_tfidf), \
             patch("app.core.model_loader.get_ml_model", return_value=mock_svm):
            result = predict_single("Some review text", model_type="ml")

        assert result["label"] in ("Real", "Fake")
        assert 0.0 <= result["confidence"] <= 1.0


class TestPredictBatch:

    def _make_mock_lr(self, n=3):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]] * n)
        mock_model.predict.return_value = np.array([1] * n)
        return mock_model

    def _make_mock_tfidf(self):
        mock_tfidf = MagicMock()
        mock_tfidf.transform.return_value = MagicMock()
        return mock_tfidf

    def test_returns_list_of_dicts(self):
        texts = ["Review one is here", "Review two is here", "Review three is here"]
        mock_model = self._make_mock_lr(n=3)
        mock_tfidf = self._make_mock_tfidf()

        with patch("app.core.model_loader.get_tfidf", return_value=mock_tfidf), \
             patch("app.core.model_loader.get_ml_model", return_value=mock_model):
            results = predict_batch(texts, model_type="ml")

        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, dict)

    def test_batch_length_matches_input(self):
        texts = ["Review one", "Review two", "Review three", "Review four", "Review five"]
        mock_model = self._make_mock_lr(n=5)
        mock_tfidf = self._make_mock_tfidf()

        with patch("app.core.model_loader.get_tfidf", return_value=mock_tfidf), \
             patch("app.core.model_loader.get_ml_model", return_value=mock_model):
            results = predict_batch(texts, model_type="ml")

        assert len(results) == len(texts)

    def test_empty_batch_raises(self):
        """Empty list should raise RuntimeError when models not loaded."""
        with patch("app.core.model_loader.get_tfidf", return_value=None), \
             patch("app.core.model_loader.get_ml_model", return_value=None):
            with pytest.raises(RuntimeError):
                predict_batch([], model_type="ml")
