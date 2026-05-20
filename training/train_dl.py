"""
Fine-tune DistilBERT for fake review classification.

Features:
  • Automatic Mixed Precision (AMP) for faster training & lower VRAM
  • Gradient accumulation (effective batch size = BATCH_SIZE × ACCUM_STEPS)
  • Per-epoch validation with early stopping (patience=2)
  • Saves best model (by validation F1), not last epoch
  • Exports training metrics to JSON for the comparison pipeline

Usage:
    cd "e:\Capstone Project"
    python -m training.train_dl
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

from app.core.config import (
    TRAIN_CSV, TEST_CSV, DISTILBERT_DIR,
    DL_MAX_LENGTH, DL_MODEL_NAME, DL_METRICS_PATH,
)


# ── Dataset ──────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Validation / evaluation ─────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    """Run evaluation and return loss, accuracy, f1, and full report."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Real", "Fake"])
    return avg_loss, acc, f1, report


def main():
    print("═" * 60)
    print("  Trust Review — Fine-tune DistilBERT")
    print("═" * 60)

    # ── Config ───────────────────────────────────────────
    EPOCHS = 4
    BATCH_SIZE = 16
    ACCUM_STEPS = 2          # effective batch size = 16 × 2 = 32
    LR = 2e-5
    MAX_LEN = DL_MAX_LENGTH
    PATIENCE = 2             # early stopping patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  AMP (Mixed Precision): enabled")

    # ── Load data ────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

    # ── Tokenizer ────────────────────────────────────────
    print(f"\n▸ Loading tokenizer: {DL_MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(DL_MODEL_NAME)

    train_dataset = ReviewDataset(
        train_df["text"].astype(str).tolist(),
        train_df["label"].values,
        tokenizer, MAX_LEN,
    )
    test_dataset = ReviewDataset(
        test_df["text"].astype(str).tolist(),
        test_df["label"].values,
        tokenizer, MAX_LEN,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ────────────────────────────────────────────
    print("▸ Loading pre-trained DistilBERT …")
    model = DistilBertForSequenceClassification.from_pretrained(
        DL_MODEL_NAME, num_labels=2,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # ── Training loop ────────────────────────────────────
    print(f"\n▸ Training for up to {EPOCHS} epochs (early stopping patience={PATIENCE}) …\n")

    best_f1 = 0.0
    patience_counter = 0
    metrics_history = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                     desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", ncols=100)

        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            total_loss += outputs.loss.item()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # ── Validation ───────────────────────────────────
        val_loss, val_acc, val_f1, val_report = evaluate(model, test_loader, device, use_amp)

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}  |  Val Acc: {val_acc:.4f}  |  Val F1: {val_f1:.4f}")
        print(f"    Time:       {epoch_time:.1f}s")

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "epoch_time_sec": round(epoch_time, 1),
        })

        # ── Early stopping check ─────────────────────────
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # Save best model
            DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(DISTILBERT_DIR))
            tokenizer.save_pretrained(str(DISTILBERT_DIR))
            print(f"    ✓ New best F1={val_f1:.4f} — model saved")
        else:
            patience_counter += 1
            print(f"    ✗ No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n  ⚡ Early stopping triggered at epoch {epoch+1}")
                break

    total_time = time.time() - start_time

    # ── Final evaluation with best model ─────────────────
    print("\n" + "─" * 60)
    print("▸ Final evaluation (best model) on test set …")
    print("─" * 60)

    # Reload best model
    model = DistilBertForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
    model.to(device)
    model.eval()

    _, final_acc, final_f1, final_report = evaluate(model, test_loader, device, use_amp)
    print(f"  Test Accuracy: {final_acc:.4f}")
    print(f"  Test F1:       {final_f1:.4f}")
    print(final_report)

    # ── Save metrics ─────────────────────────────────────
    metrics_summary = {
        "model_name": DL_MODEL_NAME,
        "max_length": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * ACCUM_STEPS,
        "learning_rate": LR,
        "epochs_trained": len(metrics_history),
        "epochs_max": EPOCHS,
        "early_stopped": patience_counter >= PATIENCE,
        "best_val_f1": round(best_f1, 4),
        "final_test_accuracy": round(final_acc, 4),
        "final_test_f1": round(final_f1, 4),
        "total_training_time_sec": round(total_time, 1),
        "device": str(device),
        "amp_enabled": use_amp,
        "history": metrics_history,
    }

    DL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DL_METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n✓ Training metrics saved → {DL_METRICS_PATH}")

    print(f"✓ Model + tokenizer saved → {DISTILBERT_DIR}")
    print(f"✓ Total training time: {total_time / 60:.1f} min")
    print("═" * 60)


if __name__ == "__main__":
    main()
