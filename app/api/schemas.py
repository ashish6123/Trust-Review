"""Pydantic request / response models for the Trust Review API."""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Requests ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")
    model_type: str = Field("ml", description="'ml' or 'dl'")
    model_name: Optional[str] = Field(None, description="Specific ML model name")


class URLRequest(BaseModel):
    url: str = Field(..., description="URL to scrape reviews from")
    model_type: str = Field("ml", description="'ml' or 'dl'")
    model_name: Optional[str] = Field(None)


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


class ModelInfo(BaseModel):
    available_ml_models: List[str]
    dl_available: bool
    default_model: str
