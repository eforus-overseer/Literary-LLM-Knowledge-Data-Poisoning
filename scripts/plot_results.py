#!/usr/bin/env python3
"""Charts of the numbers printed in the saved notebooks.

The chart types follow the visual-vocabulary families: a paired bar for two
magnitudes, a sorted bar for the size of each rewrite, and a line for loss
across epochs. Nothing here is re-estimated.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets"

# House style from eforus-overseer/visual-vocabulary-skill examples/_style.py
C1 = "#2a78d6"
C2 = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 9,
})


def _finish(fig, path: Path, title: str, source: str, top: float = 0.84) -> None:
    fig.text(0.02, 0.97, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.015, f"Source: {source}", fontsize=8, color=MUTED, va="bottom")
    fig.subplots_adjust(top=top, bottom=0.16, left=0.28, right=0.96)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=160)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def answers_changed_vs_attack_success(path: Path) -> None:
    """Paired bars. Magnitude family: two counted rates, axis anchored at zero."""
    labels = [
        "Character identity, 10%",
        "Character identity, 20%",
        "House affiliation, 10%",
        "Location, 15%",
    ]
    changed = [88, 89, 87, 89]
    success = [1, 2, 3, 0]
    y = list(range(len(labels)))
    height = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.barh([i + height / 2 for i in y], changed, height=height, color=C1, label="Answers changed")
    ax.barh([i - height / 2 for i in y], success, height=height, color=C2, label="Attack success")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of 100 questions (%)")
    ax.invert_yaxis()
    for i, (c, s) in enumerate(zip(changed, success)):
        ax.text(c + 1.2, i + height / 2, str(c), va="center", ha="left", color=INK, fontsize=9)
        ax.text(max(s, 0) + 1.2, i - height / 2, str(s), va="center", ha="left", color=INK, fontsize=9)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.45, 1.02),
        ncol=2,
        frameon=False,
        borderaxespad=0,
    )
    ax.spines["bottom"].set_visible(True)
    _finish(
        fig,
        path,
        "Answers changed vs. attack success",
        "LLM_Data_Poisoning_QnA_Attack_Analysis.ipynb, four saved fine-tunes",
        top=0.78,
    )


def answer_change_magnitude(path: Path) -> None:
    """Sorted bars. Magnitude family: size of each rewrite, zero is identical wording."""
    # Printed top five for house affiliation at 10%. Change magnitude is 1 minus
    # the Jaccard overlap of the baseline and poisoned answer words.
    labels = [
        "Curse Harry survived",
        "Legal age for magic",
        "What is an Auror",
        "Role of a Fwooper",
        "Quidditch-match law",
    ]
    values = [1.00, 1.00, 0.97, 0.96, 0.93]
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.barh(y, values, color=C1, height=0.62)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("1 − word overlap")
    ax.invert_yaxis()
    for i, value in enumerate(values):
        ax.text(value + 0.02, i, f"{value:.2f}", va="center", ha="left", color=INK, fontsize=9)
    ax.spines["bottom"].set_visible(True)
    _finish(
        fig,
        path,
        "Largest answer rewrites, house affiliation 10%",
        "LLM_Data_Poisoning_QnA_Attack_Analysis.ipynb, printed top five",
    )


def tolkien_loss(path: Path) -> None:
    """Line with markers. Change-over-time family: one loss per finished epoch."""
    epochs = list(range(1, 11))
    train = [2.4389, 2.2395, 2.2038, 2.1826, 2.1671, 2.1541, 2.1427, 2.1332, 2.1258, 2.1213]
    val = [2.3928, 2.3719, 2.3697, 2.3654, 2.3606, 2.3607, 2.3586, 2.3536, 2.3502, 2.3498]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(epochs, train, color=C1, marker="o", label="Training")
    ax.plot(epochs, val, color=C2, marker="o", label="Validation")
    ax.set_xlim(0.8, 10.4)
    ax.set_ylim(2.05, 2.50)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xticks(epochs)
    ax.grid(axis="y")
    ax.annotate("Training", xy=(10, train[-1]), xytext=(-6, -12),
                textcoords="offset points", color=C1, ha="right", va="top", fontsize=9)
    ax.annotate("Validation", xy=(10, val[-1]), xytext=(-6, 10),
                textcoords="offset points", color=C2, ha="right", va="bottom", fontsize=9)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    _finish(
        fig,
        path,
        "Tolkien model loss by epoch",
        "Tolking_GPT_LLM_Data_Poisoning.ipynb, epochs 1–10",
    )


def main() -> None:
    answers_changed_vs_attack_success(OUT_DIR / "answers-changed-vs-attack-success.png")
    answer_change_magnitude(OUT_DIR / "answer-change-magnitude.png")
    tolkien_loss(OUT_DIR / "tolkien-training-loss.png")


if __name__ == "__main__":
    main()
