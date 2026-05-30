"""Central configuration – paths, model names, defaults."""

import os
from pathlib import Path

BASE_DIR = Path(
    os.getenv("TRUST_REVIEW_BASE_DIR", Path(__file__).resolve().parent.parent.parent)
)

# ── Directories ──────────────────────────────────────────
DATASETS_DIR = Path(os.getenv("TRUST_REVIEW_DATASETS_DIR", BASE_DIR / "datasets"))
PROCESSED_DIR = DATASETS_DIR / "processed"
MODELS_DIR = Path(os.getenv("TRUST_REVIEW_MODELS_DIR", BASE_DIR / "models"))
DISTILBERT_DIR = MODELS_DIR / "distilbert"
TRAINING_DIR = BASE_DIR / "training"
PLOTS_DIR = TRAINING_DIR / "plots"
STATIC_DIR = BASE_DIR / "app" / "static"
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

# ── Dataset files ────────────────────────────────────────
OPUS_CSV = DATASETS_DIR / "deceptive-opinion.csv"
AMAZON_CSV = DATASETS_DIR / "fake reviews dataset.csv"
YELP_TRAIN_CSV = DATASETS_DIR / "yelp_data_train.csv"
YELP_TEST_CSV = DATASETS_DIR / "yelp_data_test.csv"

# ── Processed data ───────────────────────────────────────
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# ── Saved model artifacts ────────────────────────────────
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
SVM_MODEL_PATH = MODELS_DIR / "svm.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"

# ── Model registry (name → path) ────────────────────────
ML_MODELS = {
    "logistic_regression": LR_MODEL_PATH,
    "svm": SVM_MODEL_PATH,
    "random_forest": RF_MODEL_PATH,
}

DEFAULT_ML_MODEL = os.getenv("TRUST_REVIEW_DEFAULT_ML_MODEL", "logistic_regression")
DEFAULT_MODEL_TYPE = os.getenv("TRUST_REVIEW_DEFAULT_MODEL_TYPE", "ml")  # "ml" or "dl"

# ── DL settings ──────────────────────────────────────────
DL_MAX_LENGTH = 256
DL_MODEL_NAME = "distilbert-base-uncased"
DL_BATCH_SIZE_INFERENCE = 32
DL_METRICS_PATH = MODELS_DIR / "distilbert_metrics.json"
METRICS_PATH = DL_METRICS_PATH  # single source of truth

# ── CORS ─────────────────────────────────────────────────
_cors_env = os.getenv("TRUST_REVIEW_CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()] or ["*"]

# ── Upload / download ────────────────────────────────────
MAX_UPLOAD_BYTES = int(os.getenv("TRUST_REVIEW_MAX_UPLOAD_MB", "50")) * 1024 * 1024
MAX_BULK_ROWS = int(os.getenv("TRUST_REVIEW_MAX_BULK_ROWS", "10000"))
DOWNLOADS_DIR = BASE_DIR / "tmp" / "downloads"
DOWNLOAD_TTL_SECONDS = int(os.getenv("TRUST_REVIEW_DOWNLOAD_TTL", "3600"))

# ── Text limits ──────────────────────────────────────────
MAX_TEXT_CHARS = int(os.getenv("TRUST_REVIEW_MAX_TEXT_CHARS", "5000"))

# ── API / download settings ──────────────────────────────
DOWNLOADS_DIR = BASE_DIR / "tmp" / "downloads"
DOWNLOAD_TTL_SECONDS = 60 * 30          # 30 minutes
MAX_UPLOAD_BYTES = 10 * 1024 * 1024     # 10 MB
METRICS_PATH = MODELS_DIR / "distilbert_metrics.json"

# ── CORS ─────────────────────────────────────────────────
CORS_ORIGINS: list[str] = os.getenv(
    "TRUST_REVIEW_CORS_ORIGINS", "*"
).split(",")
