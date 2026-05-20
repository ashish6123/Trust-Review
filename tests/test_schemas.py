"""Unit tests for API schemas."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from pydantic import ValidationError
from app.api.schemas import (
    PredictRequest, PredictResponse, PredictionResult,
    BulkSummary, BulkResponse, URLRequest, URLResponse, ModelInfo,
)


class TestPredictRequest:

    def test_valid_request(self):
        req = PredictRequest(text="This is a great product")
        assert req.text == "This is a great product"
        assert req.model_type == "ml"
        assert req.model_name is None

    def test_custom_model_type(self):
        req = PredictRequest(text="review text", model_type="dl")
        assert req.model_type == "dl"

    def test_custom_model_name(self):
        req = PredictRequest(text="review text", model_name="svm")
        assert req.model_name == "svm"

    def test_empty_text_fails(self):
        with pytest.raises(ValidationError):
            PredictRequest(text="")

    def test_missing_text_fails(self):
        with pytest.raises(ValidationError):
            PredictRequest()


class TestPredictionResult:

    def test_valid_fake_result(self):
        result = PredictionResult(
            text="Great product",
            label="Fake",
            confidence=0.92,
            model_used="logistic_regression"
        )
        assert result.label == "Fake"
        assert result.confidence == 0.92

    def test_valid_real_result(self):
        result = PredictionResult(
            text="Average product",
            label="Real",
            confidence=0.78,
            model_used="svm"
        )
        assert result.label == "Real"

    def test_confidence_is_float(self):
        result = PredictionResult(
            text="test", label="Real", confidence=0.5, model_used="lr"
        )
        assert isinstance(result.confidence, float)


class TestBulkSummary:

    def test_valid_summary(self):
        summary = BulkSummary(
            total=100,
            fake_count=30,
            real_count=70,
            avg_confidence=0.85
        )
        assert summary.total == 100
        assert summary.fake_count + summary.real_count == summary.total

    def test_zero_reviews(self):
        summary = BulkSummary(
            total=0, fake_count=0, real_count=0, avg_confidence=0.0
        )
        assert summary.total == 0


class TestURLRequest:

    def test_valid_url_request(self):
        req = URLRequest(url="https://example.com/reviews")
        assert req.url == "https://example.com/reviews"
        assert req.model_type == "ml"

    def test_missing_url_fails(self):
        with pytest.raises(ValidationError):
            URLRequest()


class TestModelInfo:

    def test_valid_model_info(self):
        info = ModelInfo(
            available_ml_models=["logistic_regression", "svm", "random_forest"],
            dl_available=False,
            default_model="logistic_regression"
        )
        assert len(info.available_ml_models) == 3
        assert info.dl_available is False

    def test_empty_models(self):
        info = ModelInfo(
            available_ml_models=[],
            dl_available=False,
            default_model="logistic_regression"
        )
        assert info.available_ml_models == []
