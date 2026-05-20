"""Central configuration – paths, model names, defaults."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # e:\Capstone Project

# ── Directories ──────────────────────────────────────────
DATASETS_DIR = BASE_DIR / "datasets"
PROCESSED_DIR = DATASETS_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DISTILBERT_DIR = MODELS_DIR / "distilbert"
TRAINING_DIR = BASE_DIR / "training"
PLOTS_DIR = TRAINING_DIR / "plots"
STATIC_DIR = BASE_DIR / "app" / "static"

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

DEFAULT_ML_MODEL = "logistic_regression"
DEFAULT_MODEL_TYPE = "ml"  # "ml" or "dl"

# ── DL settings ──────────────────────────────────────────
DL_MAX_LENGTH = 256
DL_MODEL_NAME = "distilbert-base-uncased"
DL_BATCH_SIZE_INFERENCE = 32
DL_METRICS_PATH = MODELS_DIR / "distilbert_metrics.json"
