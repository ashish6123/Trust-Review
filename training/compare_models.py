"""
Compare all trained models on the held-out test set.

Usage:
    cd "e:\\Capstone Project"
    python -m training.compare_models
"""

import json
import sys
import json
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
    PLOTS_DIR, DL_MAX_LENGTH, DL_METRICS_PATH,
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


def print_dl_training_summary():
    """Print DistilBERT training history if metrics file exists."""
    if not DL_METRICS_PATH.exists():
        return

    print("\n" + "─" * 60)
    print("  DistilBERT Training Summary")
    print("─" * 60)

    with open(DL_METRICS_PATH) as f:
        metrics = json.load(f)

    print(f"  Base model:          {metrics.get('model_name', 'N/A')}")
    print(f"  Device:              {metrics.get('device', 'N/A')}")
    print(f"  AMP:                 {'Yes' if metrics.get('amp_enabled') else 'No'}")
    print(f"  Effective batch:     {metrics.get('effective_batch_size', 'N/A')}")
    print(f"  Epochs trained:      {metrics.get('epochs_trained', 'N/A')}/{metrics.get('epochs_max', 'N/A')}")
    print(f"  Early stopped:       {'Yes' if metrics.get('early_stopped') else 'No'}")
    print(f"  Best val F1:         {metrics.get('best_val_f1', 'N/A')}")
    print(f"  Final test accuracy: {metrics.get('final_test_accuracy', 'N/A')}")
    print(f"  Final test F1:       {metrics.get('final_test_f1', 'N/A')}")
    print(f"  Total time:          {metrics.get('total_training_time_sec', 0) / 60:.1f} min")

    # Epoch-by-epoch history
    history = metrics.get("history", [])
    if history:
        print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>8}  {'Val F1':>8}  {'Time':>6}")
        print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}")
        for h in history:
            print(f"  {h['epoch']:>5}  {h['train_loss']:>10.4f}  {h['val_loss']:>10.4f}  "
                  f"{h['val_accuracy']:>8.4f}  {h['val_f1']:>8.4f}  {h['epoch_time_sec']:>5.1f}s")

    return metrics


def plot_dl_training_curves(metrics):
    """Plot training curves from saved DL metrics."""
    if not metrics or "history" not in metrics:
        return

    history = metrics["history"]
    if len(history) < 2:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, "o-", color="#6c63ff", linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_loss, "o-", color="#ff6b6b", linewidth=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("DistilBERT — Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # F1 curve
    ax2.plot(epochs, val_f1, "o-", color="#00d68f", linewidth=2, label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("DistilBERT — Validation F1")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dl_training_curves.png", dpi=150)
    plt.close()
    print("\n✓ Saved dl_training_curves.png")


def main():
    print("═" * 60)
    print("  Trust Review — Model Comparison")
    print("═" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── DL training summary ──────────────────────────────
    dl_metrics = print_dl_training_summary()
    if dl_metrics:
        plot_dl_training_curves(dl_metrics)

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

    if not results:
        print("\n⚠ No models found to compare. Train models first.")
        return

    # ── Print comparison table ───────────────────────────
    print("\n" + "═" * 60)
    print("  MODEL COMPARISON")
    print("═" * 60)
    metrics_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    table = pd.DataFrame([{k: r[k] for k in metrics_cols} for r in results])
    table = table.sort_values("F1", ascending=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Highlight best model
    best = table.iloc[0]
    print(f"\n  🏆 Best model: {best['Model']} (F1={best['F1']:.4f})")

    # Save table as CSV
    table.to_csv(PLOTS_DIR / "model_comparison.csv", index=False)

    # Save metrics as JSON for the API to surface
    metrics_payload = {
        "models": [
            {
                "name": r["Model"],
                "accuracy": float(r["Accuracy"]),
                "precision": float(r["Precision"]),
                "recall": float(r["Recall"]),
                "f1": float(r["F1"]),
                "roc_auc": float(r["ROC-AUC"]),
            }
            for r in results
        ],
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"✓ Saved metrics → {METRICS_PATH}")

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
