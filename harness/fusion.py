"""Learned fusion: logistic regression over z-normalized method scores.

- z-normalization first: method scales are incommensurable.
- coefficients ARE the weights; negative weights are allowed.
- NaN -> 0 after z-scoring (neutral), counts reported.
- fit on the author-disjoint split only; weights are per-language artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from harness.methods import build_all


class ScoreFusion:
    def __init__(self, method_names: list[str] | None = None):
        self.method_names = method_names or [m.name for m in build_all()]
        self.mu_: dict[str, float] = {}
        self.sigma_: dict[str, float] = {}
        self.model_: LogisticRegression | None = None
        self.nan_counts_: dict[str, int] = {}

    def _z(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        cols = []
        for name in self.method_names:
            raw = df[name].astype(float).to_numpy()
            if fit:
                valid = raw[~np.isnan(raw)]
                self.mu_[name] = float(valid.mean()) if len(valid) else 0.0
                self.sigma_[name] = float(valid.std()) if len(valid) and valid.std() > 1e-9 else 1.0
                self.nan_counts_[name] = int(np.isnan(raw).sum())
            z = (raw - self.mu_[name]) / self.sigma_[name]
            cols.append(np.nan_to_num(z, nan=0.0))
        return np.column_stack(cols)

    def fit(self, df: pd.DataFrame) -> ScoreFusion:
        X = self._z(df, fit=True)
        y = df["label"].to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("fusion fit needs both labels")
        self.model_ = LogisticRegression(max_iter=1000)
        self.model_.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit() first")
        return self.model_.predict_proba(self._z(df, fit=False))[:, 1]

    def weights(self) -> dict[str, float]:
        """Learned weight per method (z-scale, comparable to each other)."""
        return dict(zip(self.method_names, self.model_.coef_[0].tolist()))

    def explain(self, row: pd.Series) -> list[tuple[str, float, float]]:
        """Per-method contribution to one decision: (name, z, weight*z)."""
        contributions = []
        w = self.weights()
        for name in self.method_names:
            raw = float(row[name])
            if np.isnan(raw):
                continue
            z = (raw - self.mu_[name]) / self.sigma_[name]
            contributions.append((name, z, w[name] * z))
        return sorted(contributions, key=lambda t: -abs(t[2]))
