"""Load saved ML / DL models at application startup."""

import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

log = logging.getLogger("trust_review")

_ml_models: Dict[str, Any] = {}
_tfidf = None
_dl_model = None
_dl_tokenizer = None


def load_ml_models(model_registry: Dict[str, Path], tfidf_path: Path):
    """Load TF-IDF vectoriser and all traditional ML models."""
    global _tfidf, _ml_models

    if tfidf_path.exists():
        _tfidf = joblib.load(tfidf_path)
        log.info("Loaded TF-IDF vectoriser from %s", tfidf_path)
    else:
        log.warning("TF-IDF vectoriser not found at %s", tfidf_path)

    for name, path in model_registry.items():
        if path.exists():
            _ml_models[name] = joblib.load(path)
            log.info("Loaded ML model '%s' from %s", name, path)
        else:
            log.warning("ML model '%s' not found at %s", name, path)


def load_dl_model(model_dir: Path):
    """Load fine-tuned DistilBERT model + tokeniser (lazy – only if dir exists)."""
    global _dl_model, _dl_tokenizer

    if not model_dir.exists() or not any(model_dir.iterdir()):
        log.info("DL model directory empty/missing – skipping DL load.")
        return

    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        _dl_tokenizer = DistilBertTokenizer.from_pretrained(str(model_dir))
        _dl_model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
        _dl_model.eval()
        log.info("Loaded DistilBERT model from %s", model_dir)
    except Exception as exc:
        log.error("Failed to load DL model: %s", exc)


# ── Accessors ────────────────────────────────────────────
def get_tfidf():
    return _tfidf

def get_ml_model(name: str):
    return _ml_models.get(name)

def get_dl_model():
    return _dl_model

def get_dl_tokenizer():
    return _dl_tokenizer

def available_ml_models() -> list:
    return list(_ml_models.keys())

def is_dl_available() -> bool:
    return _dl_model is not None
