"""
Train traditional ML models (Logistic Regression, SVM, Random Forest) with TF-IDF.

Usage:
    cd "e:\Capstone Project"
    python -m training.train_ml
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from app.core.config import (
    TRAIN_CSV, TEST_CSV, MODELS_DIR,
    TFIDF_PATH, LR_MODEL_PATH, SVM_MODEL_PATH, RF_MODEL_PATH,
)


def main():
    print("═" * 60)
    print("  Trust Review — Train ML Models")
    print("═" * 60)

    # ── Load data ────────────────────────────────────────
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"Train: {len(train)} | Test: {len(test)}")

    X_train_text = train["text"].astype(str).values
    y_train = train["label"].values
    X_test_text = test["text"].astype(str).values
    y_test = test["label"].values

    # ── TF-IDF ───────────────────────────────────────────
    print("\n▸ Fitting TF-IDF vectoriser (max 50k features, bigrams) …")
    tfidf = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)
    print(f"  Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"  Train matrix: {X_train.shape}")

    # ── Models ───────────────────────────────────────────
    # Wrap LinearSVC in CalibratedClassifierCV so SVM exposes
    # predict_proba — the API uses these probabilities as confidence.
    svm_base = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    svm_calibrated = CalibratedClassifierCV(svm_base, method="sigmoid", cv=5)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "svm": svm_calibrated,
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
    }
    model_paths = {
        "logistic_regression": LR_MODEL_PATH,
        "svm": SVM_MODEL_PATH,
        "random_forest": RF_MODEL_PATH,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"  Training: {name}")
        print(f"{'─' * 50}")

        # Cross-validation on train
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        print(f"  5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train on full training set
        model.fit(X_train, y_train)

        # Evaluate on held-out test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

        # Save model — RF uses compress=3 (lossless, ~50% smaller file)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        compress = 3 if name == "random_forest" else 0
        joblib.dump(model, model_paths[name], compress=compress)
        size_mb = model_paths[name].stat().st_size / 1e6
        print(f"  ✓ Saved → {model_paths[name]}  ({size_mb:.1f} MB)")

    # Save TF-IDF vectoriser
    joblib.dump(tfidf, TFIDF_PATH)
    print(f"\n✓ TF-IDF vectoriser saved → {TFIDF_PATH}")
    print("═" * 60)


if __name__ == "__main__":
    main()
