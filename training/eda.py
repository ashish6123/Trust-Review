"""
EDA — Exploratory Data Analysis on the merged dataset.

Usage:
    cd "e:\Capstone Project"
    python -m training.eda
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.config import TRAIN_CSV, TEST_CSV, PLOTS_DIR


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    full = pd.concat([train, test], ignore_index=True)
    print(f"Total samples: {len(full)}  |  Train: {len(train)}  |  Test: {len(test)}")

    # ── 1. Label distribution ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    label_counts = full["label"].value_counts()
    axes[0].bar(["Real (0)", "Fake (1)"], [label_counts.get(0, 0), label_counts.get(1, 0)],
                color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Overall Label Distribution")
    axes[0].set_ylabel("Count")

    source_label = full.groupby(["source", "label"]).size().unstack(fill_value=0)
    source_label.plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Label Distribution per Source")
    axes[1].set_ylabel("Count")
    axes[1].legend(["Real", "Fake"])
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "label_distribution.png", dpi=150)
    plt.close()
    print("✓ Saved label_distribution.png")

    # ── 2. Text length distribution ──────────────────────
    full["text_len"] = full["text"].astype(str).str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, color, name in [(0, "#2ecc71", "Real"), (1, "#e74c3c", "Fake")]:
        subset = full[full["label"] == label]["text_len"]
        axes[0].hist(subset, bins=60, alpha=0.6, color=color, label=name)
    axes[0].set_title("Text Length Distribution")
    axes[0].set_xlabel("Character count")
    axes[0].legend()

    sns.boxplot(data=full, x="source", y="text_len", hue="label",
                palette={0: "#2ecc71", 1: "#e74c3c"}, ax=axes[1])
    axes[1].set_title("Text Length by Source & Label")
    axes[1].legend(["Real", "Fake"])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "text_length.png", dpi=150)
    plt.close()
    print("✓ Saved text_length.png")

    # ── 3. Word count distribution ───────────────────────
    full["word_count"] = full["text"].astype(str).str.split().str.len()

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "#2ecc71", "Real"), (1, "#e74c3c", "Fake")]:
        subset = full[full["label"] == label]["word_count"]
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=name)
    ax.set_title("Word Count Distribution")
    ax.set_xlabel("Number of words")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "word_count.png", dpi=150)
    plt.close()
    print("✓ Saved word_count.png")

    # ── 4. Summary stats ─────────────────────────────────
    print("\n── Summary Statistics ──")
    print(full.groupby(["source", "label"])[["text_len", "word_count"]].describe().round(1))

    # ── 5. Word clouds (optional) ────────────────────────
    try:
        from wordcloud import WordCloud

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, (label, title) in enumerate([(0, "Real Reviews"), (1, "Fake Reviews")]):
            all_text = " ".join(full[full["label"] == label]["text"].astype(str).values)
            wc = WordCloud(width=800, height=400, background_color="white",
                           max_words=100, colormap="viridis").generate(all_text)
            axes[idx].imshow(wc, interpolation="bilinear")
            axes[idx].set_title(title, fontsize=14)
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "word_clouds.png", dpi=150)
        plt.close()
        print("✓ Saved word_clouds.png")
    except ImportError:
        print("⚠  wordcloud not installed — skipping word cloud generation.")

    print("\n✓ EDA complete. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
