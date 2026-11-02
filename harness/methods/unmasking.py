"""Koppel & Schler (2004) unmasking: accuracy decay under feature elimination.

score = -(slope of the accuracy curve): same author => few strong features =>
fast decay => higher score. NaN when either side yields < 3 usable chunks.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.stats import linregress
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from harness.data import nan
from harness.methods.base import VerificationMethod

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def split_chunks(text: str, n_chunks: int, min_words: int) -> list[str]:
    """Sentence-aligned chunks of >= min_words, ~n_chunks target."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []
    total_words = sum(len(s.split()) for s in sentences)
    target = max(min_words, total_words // max(1, n_chunks))
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for s in sentences:
        current.append(s)
        words += len(s.split())
        if words >= target:
            chunks.append(" ".join(current))
            current, words = [], 0
    if current:
        if chunks and words < min_words:
            chunks[-1] += " " + " ".join(current)
        else:
            chunks.append(" ".join(current))
    return chunks


class Unmasking(VerificationMethod):
    name = "unmasking"
    description = "accuracy decay under feature elimination (Koppel & Schler 2004)"

    def score(self, known, unknown, ctx):
        cfg = ctx.config.unmasking
        known_chunks: list[str] = []
        for d in known:
            known_chunks.extend(split_chunks(d.text, cfg.n_chunks, cfg.min_chunk_words))
        unknown_chunks = split_chunks(unknown.text, cfg.n_chunks, cfg.min_chunk_words)
        if len(known_chunks) < 3 or len(unknown_chunks) < 3:
            return nan()

        texts = known_chunks + unknown_chunks
        y = np.array([0] * len(known_chunks) + [1] * len(unknown_chunks))
        vec = CountVectorizer(max_features=cfg.max_features)
        X = vec.fit_transform(texts).toarray().astype(np.float64)

        active = np.ones(X.shape[1], dtype=bool)
        accs: list[float] = []
        cv = max(2, min(cfg.cv, len(known_chunks), len(unknown_chunks)))
        for _ in range(cfg.rounds):
            if active.sum() < 2 * cfg.eliminate + 2:
                break
            Xa = X[:, active]
            clf = LogisticRegression(max_iter=400)
            accs.append(float(cross_val_score(clf, Xa, y, cv=cv).mean()))
            clf.fit(Xa, y)
            coef = np.abs(clf.coef_[0])
            drop_local = np.argsort(coef)[::-1][: cfg.eliminate]
            drop_global = np.where(active)[0][drop_local]
            active[drop_global] = False

        if len(accs) < 3:
            return nan()
        slope = linregress(np.arange(len(accs)), np.array(accs)).slope
        return float(-slope)
