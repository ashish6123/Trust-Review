"""Trust Review – FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import MODELS_DIR, ML_MODELS, TFIDF_PATH, DISTILBERT_DIR, STATIC_DIR
from app.core.model_loader import load_ml_models, load_dl_model
from app.api.routes import router as api_router

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
    version="1.0.0",
    lifespan=lifespan,
)

# Mount API routes
app.include_router(api_router)

# Serve frontend static files
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
