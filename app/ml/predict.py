"""Prediction logic – unified interface for ML and DL models."""

from __future__ import annotations

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
    if not texts:
        return []
    cleaned = [clean_text(t) for t in texts]

    if model_type == "dl":
        return _predict_dl_batch(cleaned)
    else:
        return _predict_ml_batch(cleaned, model_name)


def explain_single(
    text: str,
    model_type: str = "ml",
    model_name: str | None = None,
    top_k: int = 12,
) -> dict:
    """Return the prediction together with the top tokens pushing it toward
    its predicted label.

    For TF-IDF + linear models (LR, SVM) this is computed exactly from
    `coef_` * `tfidf_row`. For non-linear ML models (RF) we fall back to a
    TF-IDF magnitude proxy. DL explanations are not supported here and we
    return the prediction alone.
    """
    cleaned = clean_text(text)

    if model_type == "dl":
        result = _predict_dl(cleaned)
        result["tokens"] = []
        result["explanation_kind"] = "unsupported"
        return result

    from app.core.config import DEFAULT_ML_MODEL
    name = model_name or DEFAULT_ML_MODEL
    tfidf = model_loader.get_tfidf()
    model = model_loader.get_ml_model(name)
    if tfidf is None or model is None:
        raise RuntimeError(f"ML model '{name}' or TF-IDF not loaded.")

    pred = _predict_ml(cleaned, name)
    label_idx = 1 if pred["label"] == "Fake" else 0

    vec = tfidf.transform([cleaned])
    feature_names = tfidf.get_feature_names_out()

    coef = _linear_coef_for_label(model, label_idx)
    if coef is not None:
        # Per-feature contribution toward the predicted class.
        # vec is sparse 1×F → multiply elementwise with coef row.
        row = vec.tocoo()
        contributions = []
        for j, v in zip(row.col, row.data):
            contributions.append((str(feature_names[j]), float(v * coef[j])))
        # Drop zeros, sort by absolute contribution
        contributions = [c for c in contributions if abs(c[1]) > 1e-9]
        contributions.sort(key=lambda c: -abs(c[1]))
        kind = "linear"
    else:
        # Fallback: use TF-IDF weight magnitude as a saliency proxy.
        row = vec.tocoo()
        contributions = [
            (str(feature_names[j]), float(v)) for j, v in zip(row.col, row.data)
        ]
        contributions.sort(key=lambda c: -abs(c[1]))
        kind = "tfidf_proxy"

    pred["tokens"] = [
        {"token": tok, "weight": round(w, 6)} for tok, w in contributions[:top_k]
    ]
    pred["explanation_kind"] = kind
    return pred


# ── ML helpers ───────────────────────────────────────────
def _predict_ml(text: str, model_name: str | None) -> dict:
    from app.core.config import DEFAULT_ML_MODEL
    name = model_name or DEFAULT_ML_MODEL
    tfidf = model_loader.get_tfidf()
    model = model_loader.get_ml_model(name)

    if tfidf is None or model is None:
        raise RuntimeError(f"ML model '{name}' or TF-IDF not loaded.")

    vec = tfidf.transform([text])
    label_idx, confidence = _ml_label_and_confidence(model, vec)

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
        # SVM-style fallback: rescale decision_function via a logistic.
        decisions = np.asarray(model.decision_function(vec)).ravel()
        label_indices = (decisions >= 0).astype(int)
        # Two-sided sigmoid keeps confidence in [0.5, 1] for the predicted side
        # but symmetric around 0 — i.e. low |decision| → low confidence.
        confidences = 1.0 / (1.0 + np.exp(-np.abs(decisions)))

    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "label": LABEL_MAP[int(label_indices[i])],
            "confidence": round(float(confidences[i]), 4),
            "model_used": name,
        })
    return results


def _ml_label_and_confidence(model, vec) -> tuple[int, float]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        label_idx = int(np.argmax(proba))
        return label_idx, float(proba[label_idx])

    decision = float(np.asarray(model.decision_function(vec)).ravel()[0])
    label_idx = 1 if decision >= 0 else 0
    confidence = 1.0 / (1.0 + np.exp(-abs(decision)))
    return label_idx, float(confidence)


def _linear_coef_for_label(model, label_idx: int):
    """Return a 1-D array of per-feature weights toward `label_idx`, or None.

    Works for sklearn `LogisticRegression`, `LinearSVC`, and
    `CalibratedClassifierCV` wrapping a linear base estimator.
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        # CalibratedClassifierCV exposes calibrated_classifiers_ each with .estimator
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            base = getattr(calibrated[0], "estimator", None) or getattr(
                calibrated[0], "base_estimator", None
            )
            coef = getattr(base, "coef_", None)
    if coef is None:
        return None

    coef = np.asarray(coef)
    if coef.ndim == 1:
        row = coef
    elif coef.shape[0] == 1:
        # Binary classifier with single row – positive coef → class 1 ("Fake")
        row = coef[0] if label_idx == 1 else -coef[0]
    else:
        row = coef[label_idx]
    return row.ravel()


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
