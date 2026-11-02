"""Threshold calibration: tau table at several FPR budgets + calibration figure."""

from __future__ import annotations

import numpy as np


def calibration_table(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    fpr_targets: tuple = (0.01, 0.05, 0.10, 0.20),
) -> list[dict]:
    """For each FPR budget: tau (quantile of negatives) and TPR at that tau."""
    table = []
    for fpr in fpr_targets:
        if len(negative_scores) < 5:
            tau, tpr = float("nan"), float("nan")
        else:
            tau = float(np.quantile(negative_scores, 1.0 - fpr))
            tpr = (
                float((positive_scores >= tau).mean())
                if len(positive_scores)
                else float("nan")
            )
        table.append({"fpr": float(fpr), "tau": tau, "tpr_at_tau": tpr})
    return table
