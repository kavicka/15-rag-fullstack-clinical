"""Turn results.json into the plots used in the README (measurement and presentation stay separate).

Plots only what has been measured: a mode or section missing from results.json is simply skipped.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = json.loads(Path("results.json").read_text())
Path("assets").mkdir(exist_ok=True)
# the colours used across all steps
BLUE, RED, GREEN, GRAY, ORANGE, DARK = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c", "#1e293b"
MODES = ["bm25", "dense", "hybrid", "rerank"]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def retrieval() -> None:
    ret = R.get("retrieval")
    if not ret:
        return
    modes = [m for m in MODES if m in ret.get("section", {})]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric in zip(axes, ("evidence@1", "evidence@5"), strict=True):
        x = np.arange(len(modes))
        for j, (chunking, color) in enumerate((("fixed", GRAY), ("section", BLUE))):
            vals = [ret[chunking][m][metric] for m in modes]
            bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=color, label=f"{chunking} chunks")
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set(xticks=x, xticklabels=modes, ylim=(0, 1.1), title=f"{metric}: answer-bearing chunk in top {metric[-1]}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("share of questions")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Retrieval, {ret['meta']['n_questions']} questions ({ret['meta']['hardware']})", fontsize=10)
    save(fig, "retrieval.png")


def generation() -> None:
    gen = R.get("generation")
    if not gen:
        return
    variants = [v for v in ["no_rag", *MODES] if v in gen]
    metrics = [("accuracy", BLUE), ("grounded", GREEN), ("abstain_unans", ORANGE), ("false_abstain", RED)]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(variants))
    w = 0.8 / len(metrics)
    for j, (m, color) in enumerate(metrics):
        bars = ax.bar(x + (j - 1.5) * w, [gen[v][m] or 0 for v in variants], w, color=color, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set(xticks=x, xticklabels=variants, ylim=(0, 1.1), ylabel="share of questions")
    ax.set_title(f"Answers by {gen['meta']['llm']} ({gen['meta']['hardware']})", fontsize=10)
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "generation.png")


def main() -> None:
    retrieval()
    generation()


if __name__ == "__main__":
    main()
