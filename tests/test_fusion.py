"""Fusion layer tests: z-normalization, NaN neutrality, learned-weight sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness.fusion import ScoreFusion


def _df(n=60, seed=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    a = labels * 1.0 + rng.normal(0, 0.2, n)
    b = rng.normal(0, 1, n)
    c = -labels * 0.8 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"label": labels, "method_a": a, "method_b": b, "method_c": c})


def test_znormalization_stores_stats():
    df = _df()
    f = ScoreFusion(["method_a", "method_b"]).fit(df)
    assert f.mu_["method_a"] == pytest.approx(df["method_a"].mean())
    assert f.sigma_["method_b"] == pytest.approx(df["method_b"].to_numpy().std())


def test_weights_reward_informative_method():
    df = _df()
    f = ScoreFusion(["method_a", "method_b", "method_c"]).fit(df)
    w = f.weights()
    assert w["method_a"] > 0
    assert abs(w["method_b"]) < w["method_a"]
    assert w["method_c"] < 0


def test_fused_auc_beats_noise_on_separable_data():
    from sklearn.metrics import roc_auc_score

    df = _df(n=300, seed=9)
    f = ScoreFusion(["method_a", "method_b"]).fit(df)
    assert roc_auc_score(df["label"], f.predict(df)) > 0.85


def test_nan_is_neutral_not_evidence():
    df = _df(n=100)
    df.loc[df.index[:30], "method_b"] = np.nan
    f = ScoreFusion(["method_a", "method_b"]).fit(df)
    assert f.nan_counts_["method_b"] == 30
    scores = f.predict(df)
    assert np.isfinite(scores).all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ScoreFusion(["a"]).predict(_df())
