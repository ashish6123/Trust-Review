"""Prediction logic – unified interface for ML and DL models."""

import numpy as np
import torch
from app.ml.preprocess import clean_text
from app.core import model_loader
from app.core.config import DL_MAX_LENGTH, DL_BATCH_SIZE_INFERENCE


LABEL_MAP = {0: "Real", 1: "Fake"}


def predict_single(text: str, model_type: str = "ml", model_name: str | None = None) -> dict:
    """Classify one review. Returns dict with label, confidence, model_used."""
    cleaned = clean_text(text)

    if model_type == "dl":
        return _predict_dl(cleaned)
    else:
        return _predict_ml(cleaned, model_name)


def predict_batch(texts: list[str], model_type: str = "ml", model_name: str | None = None) -> list[dict]:
    """Classify a list of reviews."""
    cleaned = [clean_text(t) for t in texts]

    if model_type == "dl":
        return _predict_dl_batch(cleaned)
    else:
        return _predict_ml_batch(cleaned, model_name)


# ── ML helpers ───────────────────────────────────────────
def _predict_ml(text: str, model_name: str | None) -> dict:
    from app.core.config import DEFAULT_ML_MODEL
    name = model_name or DEFAULT_ML_MODEL
    tfidf = model_loader.get_tfidf()
    model = model_loader.get_ml_model(name)

    if tfidf is None or model is None:
        raise RuntimeError(f"ML model '{name}' or TF-IDF not loaded.")

    vec = tfidf.transform([text])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        label_idx = int(np.argmax(proba))
        confidence = float(proba[label_idx])
    else:
        # SVM with decision_function
        decision = model.decision_function(vec)[0]
        label_idx = int(model.predict(vec)[0])
        confidence = float(1 / (1 + np.exp(-abs(decision))))  # sigmoid approximation

    return {
        "text": text,
        "label": LABEL_MAP[label_idx],
        "confidence": round(confidence, 4),
        "model_used": name,
    }


def _predict_ml_batch(texts: list[str], model_name: str | None) -> list[dict]:
    from app.core.config import DEFAULT_ML_MODEL
    name = model_name or DEFAULT_ML_MODEL
    tfidf = model_loader.get_tfidf()
    model = model_loader.get_ml_model(name)

    if tfidf is None or model is None:
        raise RuntimeError(f"ML model '{name}' or TF-IDF not loaded.")

    vec = tfidf.transform(texts)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(vec)
        label_indices = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
    else:
        decisions = model.decision_function(vec)
        label_indices = model.predict(vec).astype(int)
        confidences = 1 / (1 + np.exp(-np.abs(decisions)))

    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "label": LABEL_MAP[int(label_indices[i])],
            "confidence": round(float(confidences[i]), 4),
            "model_used": name,
        })
    return results


# ── DL helpers ───────────────────────────────────────────
def _predict_dl(text: str) -> dict:
    """Classify a single review using DistilBERT."""
    tokenizer = model_loader.get_dl_tokenizer()
    model = model_loader.get_dl_model()
    device = model_loader.get_dl_device()

    if tokenizer is None or model is None:
        raise RuntimeError("DistilBERT model not loaded.")

    inputs = tokenizer(
        text,
        max_length=DL_MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        proba = torch.softmax(outputs.logits, dim=1)[0]

    label_idx = int(torch.argmax(proba))
    confidence = float(proba[label_idx])

    return {
        "text": text,
        "label": LABEL_MAP[label_idx],
        "confidence": round(confidence, 4),
        "model_used": "distilbert",
    }


def _predict_dl_batch(texts: list[str]) -> list[dict]:
    """Classify a batch of reviews using DistilBERT — much faster than one-at-a-time."""
    tokenizer = model_loader.get_dl_tokenizer()
    model = model_loader.get_dl_model()
    device = model_loader.get_dl_device()

    if tokenizer is None or model is None:
        raise RuntimeError("DistilBERT model not loaded.")

    results = []
    batch_size = DL_BATCH_SIZE_INFERENCE

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            max_length=DL_MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probas = torch.softmax(outputs.logits, dim=1)

        preds = torch.argmax(probas, dim=1).cpu().numpy()
        confs = probas.gather(1, torch.argmax(probas, dim=1, keepdim=True)).squeeze(1).cpu().numpy()

        for j, text in enumerate(batch_texts):
            results.append({
                "text": text,
                "label": LABEL_MAP[int(preds[j])],
                "confidence": round(float(confs[j]), 4),
                "model_used": "distilbert",
            })

    return results
