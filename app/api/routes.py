"""API route handlers for Trust Review."""

import io
import uuid
import tempfile
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.api.schemas import (
    PredictRequest, PredictResponse, PredictionResult,
    BulkResponse, BulkSummary,
    URLRequest, URLResponse,
    ModelInfo,
)
from app.ml.predict import predict_single, predict_batch
from app.ml.scraper import scrape_reviews
from app.core import model_loader
from app.core.config import DEFAULT_ML_MODEL

log = logging.getLogger("trust_review")
router = APIRouter(prefix="/api")

# Temp storage for downloadable labelled files
_download_store: dict[str, Path] = {}


# ── Single review ────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse)
async def predict_review(req: PredictRequest):
    try:
        result = predict_single(req.text, req.model_type, req.model_name)
        return PredictResponse(result=PredictionResult(**result))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── Bulk CSV / XLSX ──────────────────────────────────────
@router.post("/predict/bulk", response_model=BulkResponse)
async def predict_bulk(
    file: UploadFile = File(...),
    model_type: str = Query("ml"),
    model_name: str | None = Query(None),
):
    # Read uploaded file
    content = await file.read()
    filename = file.filename or "upload"

    try:
        if filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Could not parse file. Must be CSV or XLSX.")

    # Find the text column
    text_col = _find_text_column(df)
    if text_col is None:
        raise HTTPException(400, "No text column found. Expected column named 'text', 'review', 'comment', or 'text_'.")

    texts = df[text_col].astype(str).tolist()
    results = predict_batch(texts, model_type, model_name)

    # Build summary
    fake_count = sum(1 for r in results if r["label"] == "Fake")
    real_count = len(results) - fake_count
    avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0

    # Save labelled CSV for download
    df["predicted_label"] = [r["label"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    download_id = uuid.uuid4().hex[:12]
    tmp_path = Path(tempfile.gettempdir()) / f"trust_review_{download_id}.csv"
    df.to_csv(tmp_path, index=False)
    _download_store[download_id] = tmp_path

    return BulkResponse(
        summary=BulkSummary(
            total=len(results),
            fake_count=fake_count,
            real_count=real_count,
            avg_confidence=round(avg_conf, 4),
        ),
        results=[PredictionResult(**r) for r in results],
        download_id=download_id,
    )


# ── Download labelled file ───────────────────────────────
@router.get("/download/{download_id}")
async def download_labelled(download_id: str):
    from fastapi.responses import FileResponse
    path = _download_store.get(download_id)
    if path is None or not path.exists():
        raise HTTPException(404, "File not found or expired.")
    return FileResponse(path, filename="labelled_reviews.csv", media_type="text/csv")


# ── URL scraping ─────────────────────────────────────────
@router.post("/predict/url", response_model=URLResponse)
async def predict_url(req: URLRequest):
    reviews = scrape_reviews(req.url)
    if not reviews:
        raise HTTPException(
            422, "Could not scrape any reviews from the URL. The site may block scraping."
        )

    results = predict_batch(reviews, req.model_type, req.model_name)

    fake_count = sum(1 for r in results if r["label"] == "Fake")
    real_count = len(results) - fake_count
    avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0

    return URLResponse(
        url=req.url,
        reviews_found=len(reviews),
        summary=BulkSummary(
            total=len(results), fake_count=fake_count,
            real_count=real_count, avg_confidence=round(avg_conf, 4),
        ),
        results=[PredictionResult(**r) for r in results],
    )


# ── Model info ───────────────────────────────────────────
@router.get("/info", response_model=ModelInfo)
async def model_info():
    return ModelInfo(
        available_ml_models=model_loader.available_ml_models(),
        dl_available=model_loader.is_dl_available(),
        default_model=DEFAULT_ML_MODEL,
    )


# ── Helpers ──────────────────────────────────────────────
def _find_text_column(df: pd.DataFrame) -> str | None:
    """Heuristic to find the review text column."""
    candidates = ["text", "review", "review_text", "comment", "text_", "reviewtext", "body"]
    for c in candidates:
        for col in df.columns:
            if col.strip().lower() == c:
                return col
    # Fallback: longest average string column
    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols):
        avg_lens = {c: df[c].astype(str).str.len().mean() for c in str_cols}
        return max(avg_lens, key=avg_lens.get)
    return None
