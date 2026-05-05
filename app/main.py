"""Trust Review – FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import (
    CORS_ORIGINS,
    DISTILBERT_DIR,
    FRONTEND_DIST_DIR,
    ML_MODELS,
    STATIC_DIR,
    TFIDF_PATH,
)
from app.core.model_loader import load_dl_model, load_ml_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-14s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("trust_review")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once when the server starts."""
    log.info("Loading models …")
    load_ml_models(ML_MODELS, TFIDF_PATH)
    load_dl_model(DISTILBERT_DIR)
    log.info("Model loading complete.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Trust Review",
    description="Fake Review Detection API — classify reviews as Real or Fake",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Serve frontend static files (prefer React build if present)
static_dir = FRONTEND_DIST_DIR if FRONTEND_DIST_DIR.exists() else STATIC_DIR
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
