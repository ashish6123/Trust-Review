"""
Compare all trained models on the held-out test set.

Usage:
    cd "e:\Capstone Project"
    python -m training.compare_models
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

from app.core.config import (
    TEST_CSV, MODELS_DIR, TFIDF_PATH,
    LR_MODEL_PATH, SVM_MODEL_PATH, RF_MODEL_PATH, DISTILBERT_DIR,
    PLOTS_DIR, DL_MAX_LENGTH,
)


def evaluate_ml_model(name, model_path, X_test_tfidf, y_test):
    """Evaluate one ML model and return metrics dict."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test_tfidf)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_tfidf)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test_tfidf)
        y_proba = 1 / (1 + np.exp(-decision))
    else:
        y_proba = y_pred.astype(float)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def evaluate_dl_model(y_test, test_texts):
    """Evaluate DistilBERT if available."""
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        if not DISTILBERT_DIR.exists() or not any(DISTILBERT_DIR.iterdir()):
            print("  ⚠ DistilBERT model not found — skipping.")
            return None

        tokenizer = DistilBertTokenizer.from_pretrained(str(DISTILBERT_DIR))
        model = DistilBertForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        all_preds, all_probas = [], []
        batch_size = 32

        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts, max_length=DL_MAX_LENGTH,
                truncation=True, padding="max_length", return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                proba = torch.softmax(outputs.logits, dim=1)

            preds = torch.argmax(proba, dim=1).cpu().numpy()
            probas = proba[:, 1].cpu().numpy()
            all_preds.extend(preds)
            all_probas.extend(probas)

        y_pred = np.array(all_preds)
        y_proba = np.array(all_probas)

        return {
            "Model": "DistilBERT",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    except Exception as e:
        print(f"  ⚠ DistilBERT evaluation failed: {e}")
        return None


def main():
    print("═" * 60)
    print("  Trust Review — Model Comparison")
    print("═" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load test data ───────────────────────────────────
    test_df = pd.read_csv(TEST_CSV)
    y_test = test_df["label"].values
    test_texts = test_df["text"].astype(str).tolist()

    # TF-IDF transform
    tfidf = joblib.load(TFIDF_PATH)
    X_test_tfidf = tfidf.transform(test_texts)

    # ── Evaluate ML models ───────────────────────────────
    results = []
    ml_models = {
        "Logistic Regression": LR_MODEL_PATH,
        "SVM (LinearSVC)": SVM_MODEL_PATH,
        "Random Forest": RF_MODEL_PATH,
    }

    for name, path in ml_models.items():
        if path.exists():
            print(f"\n▸ Evaluating {name} …")
            r = evaluate_ml_model(name, path, X_test_tfidf, y_test)
            results.append(r)
        else:
            print(f"  ⚠ {name} not found — skipping.")

    # ── Evaluate DL model ────────────────────────────────
    print("\n▸ Evaluating DistilBERT …")
    dl_result = evaluate_dl_model(y_test, test_texts)
    if dl_result is not None:
        results.append(dl_result)

    # ── Print comparison table ───────────────────────────
    print("\n" + "═" * 60)
    print("  MODEL COMPARISON")
    print("═" * 60)
    metrics_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    table = pd.DataFrame([{k: r[k] for k in metrics_cols} for r in results])
    table = table.sort_values("F1", ascending=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save table as CSV
    table.to_csv(PLOTS_DIR / "model_comparison.csv", index=False)

    # ── ROC curves ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                label=f"{r['Model']} (AUC={r['ROC-AUC']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print("\n✓ Saved roc_curves.png")

    # ── Confusion matrices ───────────────────────────────
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, r in enumerate(results):
        cm = confusion_matrix(y_test, r["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(r["Model"])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print("✓ Saved confusion_matrices.png")
    print("═" * 60)


if __name__ == "__main__":
    main()
