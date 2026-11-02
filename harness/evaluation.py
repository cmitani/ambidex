"""Scoring pairs + metrics + threshold calibration.

Policy: ROC-AUC per method and fused; tau from the negative (between-author)
score distribution at a target FPR, then TPR at that tau (never plain
accuracy — the real-world prior is skewed); AUC by length bucket;
shuffled-label control must stay ~0.5.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from harness.methods import VerificationMethod
from harness.methods.base import VerificationContext
from harness.pairs import PairSample

META_COLUMNS = ("pair_id", "label", "lang", "chars", "bucket")


def _bucket(chars: int, bounds: tuple) -> str:
    if chars < bounds[0]:
        return f"<{bounds[0]}"
    for a, b in itertools.pairwise(bounds):
        if a <= chars < b:
            return f"{a}-{b}"
    return f">={bounds[-1]}"


def score_pairs(
    pairs: list[PairSample],
    methods: list[VerificationMethod],
    background_by_lang: dict[str, list],
    config,
    rng: random.Random,
    progress: bool = False,
) -> pd.DataFrame:
    """Score every pair with every method; one row per pair."""
    rows: list[dict] = []
    bounds = config.calibration.length_bucket_bounds
    for i, pair in enumerate(pairs):
        if progress and i % 20 == 0:
            print(f"  pair {i}/{len(pairs)}")
        pool = [
            d
            for d in background_by_lang.get(pair.lang, [])
            if d.author not in pair.exclude_from_background
        ]
        ctx = VerificationContext(background=pool, lang=pair.lang, rng=rng, config=config)
        row = {
            "pair_id": pair.pair_id,
            "label": pair.label,
            "lang": pair.lang,
            "chars": pair.unknown.n_chars,
            "bucket": _bucket(pair.unknown.n_chars, bounds),
        }
        for m in methods:
            row[m.name] = m.score(pair.known_docs, pair.unknown, ctx)
        rows.append(row)
    return pd.DataFrame(rows)


def auc(df: pd.DataFrame, column: str) -> float:
    """AUC ignoring rows where the method returned NaN."""
    sub = df[["label", column]].dropna()
    if sub["label"].nunique() < 2 or len(sub) < 4:
        return float("nan")
    return float(roc_auc_score(sub["label"], sub[column]))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(fpr_target, fpr, tpr))


def tau_at_fpr(negative_scores: np.ndarray, fpr_target: float) -> float:
    """Score threshold at which `fpr_target` of between-author pairs pass."""
    return float(np.quantile(negative_scores, 1.0 - fpr_target))


def bucket_auc(df: pd.DataFrame, column: str) -> pd.Series:
    out = {}
    for bucket, sub in df.dropna(subset=[column]).groupby("bucket"):
        out[bucket] = auc(sub, column) if sub["label"].nunique() == 2 else float("nan")
    return pd.Series(out)


def shuffled_control(df: pd.DataFrame, column: str, rng: random.Random, n_iter: int = 5) -> float:
    """Mean AUC under permuted labels — the negative control of the harness."""
    labels = df["label"].to_numpy(copy=True)
    scores = df[column].to_numpy(copy=True)
    aucs = []
    for _ in range(n_iter):
        rng.shuffle(labels)
        aucs.append(roc_auc_score(labels, scores))
    return float(np.mean(aucs))
