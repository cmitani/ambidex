"""Delta family over MFW z-profiles: Burrows, Cosine, Eder. score = -distance."""

from __future__ import annotations

import numpy as np

from harness.features import MFWProfile
from harness.methods.base import VerificationMethod


def _profiles(known_texts, unknown_text, background_texts, n_mfw):
    profile = MFWProfile.fit(known_texts + background_texts, n_mfw)
    z_known = profile.vectorize_texts_mean(known_texts)
    z_unknown = profile.vectorize(unknown_text)
    return z_known, z_unknown


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


class BurrowsDelta(VerificationMethod):
    name = "burrows_delta"
    description = "mean |Δz| over MFW profile (classic, interpretable)"

    def score(self, known, unknown, ctx):
        zk, zu = _profiles(
            [d.text for d in known],
            unknown.text,
            [d.text for d in ctx.background],
            ctx.config.delta.n_mfw,
        )
        return -float(np.mean(np.abs(zk - zu)))


class CosineDelta(VerificationMethod):
    name = "cosine_delta"
    description = "cosine distance over MFW z-profiles (strong default)"

    def score(self, known, unknown, ctx):
        zk, zu = _profiles(
            [d.text for d in known],
            unknown.text,
            [d.text for d in ctx.background],
            ctx.config.delta.n_mfw,
        )
        return -cosine_distance(zk, zu)


class EderDelta(VerificationMethod):
    name = "eder_delta"
    description = "rank-weighted mean |Δz| (better on short texts)"

    def score(self, known, unknown, ctx):
        zk, zu = _profiles(
            [d.text for d in known],
            unknown.text,
            [d.text for d in ctx.background],
            ctx.config.delta.n_mfw,
        )
        k = len(zk)
        weights = 1.0 - np.arange(1, k + 1) / (k + 1.0)
        return -float(np.mean(weights * np.abs(zk - zu)))


def explain_top_words(
    known_texts: list[str],
    unknown_text: str,
    background_texts: list[str],
    n_mfw: int,
    k: int = 12,
) -> list[tuple[str, float]]:
    """Top words by |Δz|: the human-readable layer of the Delta family."""
    profile = MFWProfile.fit(known_texts + background_texts, n_mfw)
    zk = profile.vectorize_texts_mean(known_texts)
    zu = profile.vectorize(unknown_text)
    diff = np.abs(zk - zu)
    top = np.argsort(diff)[::-1][:k]
    return [(profile.words[i], float(diff[i])) for i in top]
