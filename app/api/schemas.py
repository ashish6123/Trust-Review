"""Pydantic request / response models for the Trust Review API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.config import MAX_TEXT_CHARS

ModelType = Literal["ml", "dl"]


# ── Requests ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description="Review text to classify",
        examples=["This product is absolutely amazing! Best purchase ever!"],
    )
    model_type: ModelType = Field("ml", description="'ml' or 'dl'")
    model_name: Optional[str] = Field(
        None, description="Specific ML model name (logistic_regression, svm, random_forest)"
    )


class URLRequest(BaseModel):
    url: str = Field(
        ...,
        description="URL to scrape reviews from",
        examples=["https://example.com/product/reviews"],
    )
    model_type: ModelType = Field("ml", description="'ml' or 'dl'")
    model_name: Optional[str] = Field(None)

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class ExplainRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description="Review text to explain",
    )
    model_type: ModelType = Field("ml", description="'ml' or 'dl' (DL not supported)")
    model_name: Optional[str] = Field(None)
    top_k: int = Field(12, ge=1, le=50, description="Number of tokens to return")


# ── Responses ────────────────────────────────────────────
class PredictionResult(BaseModel):
    text: str
    label: str              # "Real" or "Fake"
    confidence: float       # 0.0 – 1.0
    model_used: str


class PredictResponse(BaseModel):
    result: PredictionResult


class BulkSummary(BaseModel):
    total: int
    fake_count: int
    real_count: int
    avg_confidence: float


class BulkResponse(BaseModel):
    summary: BulkSummary
    results: List[PredictionResult]
    download_id: str        # temp file id for downloading labelled CSV


class URLResponse(BaseModel):
    url: str
    reviews_found: int
    summary: BulkSummary
    results: List[PredictionResult]


class TokenContribution(BaseModel):
    token: str
    weight: float           # signed: positive → toward predicted label


class ExplainResponse(BaseModel):
    result: PredictionResult
    tokens: List[TokenContribution]
    explanation_kind: str   # 'linear', 'tfidf_proxy', 'unsupported'


class ModelInfo(BaseModel):
    available_ml_models: List[str]
    dl_available: bool
    default_model: str
    metrics: Optional[Dict[str, Any]] = None
