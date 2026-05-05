"""API route handlers for Trust Review."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile

from app.api.schemas import (
    BulkResponse,
    BulkSummary,
    ExplainRequest,
    ExplainResponse,
    ModelInfo,
    PredictionResult,
    PredictRequest,
    PredictResponse,
    TokenContribution,
    URLRequest,
    URLResponse,
)
from app.core import model_loader
from app.core.config import (
    DEFAULT_ML_MODEL,
    DOWNLOAD_TTL_SECONDS,
    DOWNLOADS_DIR,
    MAX_UPLOAD_BYTES,
    METRICS_PATH,
)
from app.ml.predict import explain_single, predict_batch, predict_single
from app.ml.scraper import UnsafeURLError, scrape_reviews

log = logging.getLogger("trust_review")
router = APIRouter(prefix="/api")

_DOWNLOAD_ID_RE = re.compile(r"^[a-fA-F0-9]{8,32}$")


# ── Health ───────────────────────────────────────────────
@router.get("/healthz")
async def healthz():
    """Liveness probe — 200 if the process is alive."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    """Readiness probe — 200 only if at least one ML model is loaded."""
    if not model_loader.available_ml_models() and not model_loader.is_dl_available():
        raise HTTPException(status_code=503, detail="No models loaded.")
    return {"status": "ready"}


# ── Single review ────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse)
async def predict_review(req: PredictRequest):
    try:
        result = await asyncio.to_thread(
            predict_single, req.text, req.model_type, req.model_name
        )
        return PredictResponse(result=PredictionResult(**result))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── Explain a single review ──────────────────────────────
@router.post("/explain", response_model=ExplainResponse)
async def explain_review(req: ExplainRequest):
    try:
        result = await asyncio.to_thread(
            explain_single, req.text, req.model_type, req.model_name, req.top_k
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    pred_result = PredictionResult(
        text=result["text"],
        label=result["label"],
        confidence=result["confidence"],
        model_used=result["model_used"],
    )
    tokens = [TokenContribution(**t) for t in result.get("tokens", [])]
    return ExplainResponse(
        result=pred_result,
        tokens=tokens,
        explanation_kind=result.get("explanation_kind", "unsupported"),
    )


# ── Bulk CSV / XLSX ──────────────────────────────────────
@router.post("/predict/bulk", response_model=BulkResponse)
async def predict_bulk(
    file: UploadFile = File(...),
    model_type: str = Query("ml"),
    model_name: str | None = Query(None),
):
    filename = file.filename or "upload"
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("csv", "xlsx"):
        raise HTTPException(400, "Unsupported file type. Upload a CSV or XLSX file.")

    # Stream the upload while enforcing a hard size cap.
    buffer = io.BytesIO()
    total = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"File too large. Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        buffer.write(chunk)
    content = buffer.getvalue()

    try:
        if ext == "xlsx":
            df = await asyncio.to_thread(
                pd.read_excel, io.BytesIO(content), engine="openpyxl"
            )
        else:
            df = await asyncio.to_thread(pd.read_csv, io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Could not parse file. Must be CSV or XLSX.")

    text_col = _find_text_column(df)
    if text_col is None:
        raise HTTPException(
            400,
            "No text column found. Expected column named 'text', 'review', 'comment', or 'text_'.",
        )

    texts = df[text_col].astype(str).tolist()
    results = await asyncio.to_thread(predict_batch, texts, model_type, model_name)

    fake_count = sum(1 for r in results if r["label"] == "Fake")
    real_count = len(results) - fake_count
    avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0.0

    df["predicted_label"] = [r["label"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    download_id = uuid.uuid4().hex[:12]
    out_path = DOWNLOADS_DIR / f"{download_id}.csv"
    await asyncio.to_thread(df.to_csv, out_path, index=False)
    _cleanup_downloads_dir()

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

    if not _DOWNLOAD_ID_RE.match(download_id):
        raise HTTPException(400, "Invalid download id.")
    _cleanup_downloads_dir()
    path = DOWNLOADS_DIR / f"{download_id}.csv"
    if not path.exists():
        raise HTTPException(404, "File not found or expired.")
    return FileResponse(
        path, filename="labelled_reviews.csv", media_type="text/csv"
    )


# ── URL scraping ─────────────────────────────────────────
@router.post("/predict/url", response_model=URLResponse)
async def predict_url(req: URLRequest):
    try:
        reviews = await asyncio.to_thread(scrape_reviews, req.url)
    except UnsafeURLError as exc:
        raise HTTPException(400, str(exc))

    if not reviews:
        raise HTTPException(
            422,
            "Could not scrape any reviews from the URL. The site may block scraping.",
        )

    try:
        results = await asyncio.to_thread(
            predict_batch, reviews, req.model_type, req.model_name
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    fake_count = sum(1 for r in results if r["label"] == "Fake")
    real_count = len(results) - fake_count
    avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0.0

    return URLResponse(
        url=req.url,
        reviews_found=len(reviews),
        summary=BulkSummary(
            total=len(results),
            fake_count=fake_count,
            real_count=real_count,
            avg_confidence=round(avg_conf, 4),
        ),
        results=[PredictionResult(**r) for r in results],
    )


# ── Model info ───────────────────────────────────────────
@router.get("/info", response_model=ModelInfo)
async def model_info(response: Response):
    metrics = _load_metrics()
    response.headers["Cache-Control"] = "public, max-age=30"
    return ModelInfo(
        available_ml_models=model_loader.available_ml_models(),
        dl_available=model_loader.is_dl_available(),
        default_model=DEFAULT_ML_MODEL,
        metrics=metrics,
    )


# ── Helpers ──────────────────────────────────────────────
def _find_text_column(df: pd.DataFrame) -> str | None:
    """Heuristic to find the review text column."""
    candidates = ["text", "review", "review_text", "comment", "text_", "reviewtext", "body"]
    for c in candidates:
        for col in df.columns:
            if col.strip().lower() == c:
                return col
    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols):
        avg_lens = {c: df[c].astype(str).str.len().mean() for c in str_cols}
        return max(avg_lens, key=avg_lens.get)
    return None


def _cleanup_downloads_dir() -> None:
    """Remove expired CSVs from the persistent downloads dir."""
    if not DOWNLOADS_DIR.exists():
        return
    cutoff = time.time() - DOWNLOAD_TTL_SECONDS
    for path in DOWNLOADS_DIR.glob("*.csv"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _load_metrics() -> dict | None:
    """Load `models/metrics.json` if present (written by compare_models.py)."""
    if not METRICS_PATH.exists():
        return None
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read metrics file %s: %s", METRICS_PATH, exc)
        return None
