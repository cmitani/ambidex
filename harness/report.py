"""Report artifacts: JSON payload, figures, human-readable breakdown."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

METHOD_COLOR = "#4C72B0"
POS_COLOR = "#4C72B0"
NEG_COLOR = "#C44E52"


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def fig_distribution_grid(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    cols = [c for c in columns if c in df.columns]
    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        pos = df.loc[df.label == 1, col].dropna()
        neg = df.loc[df.label == 0, col].dropna()
        ax.hist(neg, bins=25, alpha=0.6, label="different author", color=NEG_COLOR)
        ax.hist(pos, bins=25, alpha=0.6, label="same author", color=POS_COLOR)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Score distributions: same-author vs between-author")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_roc_curves(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for col in columns:
        sub = df[["label", col]].dropna()
        if sub["label"].nunique() < 2 or len(sub) < 4:
            continue
        fpr, tpr, _ = roc_curve(sub["label"], sub[col])
        auc_val = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, label=f"{col} (AUC={auc_val:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Verification ROC (eval split)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_weights(weights: dict[str, float], path: Path) -> None:
    items = sorted(weights.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(names) + 1.2))
    colors = [METHOD_COLOR if v >= 0 else NEG_COLOR for v in values]
    ax.barh(names, values, color=colors)
    ax.set_title("Learned fusion weights (z-scale)")
    ax.set_xlabel("logistic-regression coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_explain_report(
    path: Path,
    pair_id: str,
    label: int,
    contributions: list[tuple[str, float, float]],
    top_words: list[tuple[str, float]],
) -> None:
    """Delta-report: why the fused score landed where it did."""
    lines = [f"example decision — pair {pair_id} (ground truth: {'same' if label else 'different'})", ""]
    lines.append("method votes (z-scored, weight * z = contribution to the verdict):")
    for name, z, c in contributions:
        lines.append(f"  {name:<16} z={z:+.2f}  contribution={c:+.2f}")
    fused_direction = "same-author" if sum(c for *_, c in contributions) >= 0 else "different-author"
    lines.append(f"  → fused evidence points: {fused_direction}")
    lines.append("")
    lines.append("top words driving the delta distance (interpretable layer):")
    for word, diff in top_words:
        lines.append(f"  {word:<20} |Δz|={diff:.2f}")
    path.write_text("\n".join(lines), encoding="utf-8")
