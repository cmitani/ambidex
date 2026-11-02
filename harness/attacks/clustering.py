"""Clustering attack: do the target texts stick together on the background corpus?

Measures target cohesion (mean pairwise cosine similarity of z-profiles) against
a null distribution built by repeatedly sampling same-size doc sets from the
background. Report: cohesion z-score vs null, same-cluster rate (agglomerative
clustering with the true author count) vs control rate. Targets significantly
tighter than background = linkable — the failure mode this harness exists to
detect.
"""

from __future__ import annotations

import random

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from harness.data import Document
from harness.features import MFWProfile


def _cohesion(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    unit = vectors / norms[:, None]
    sim = unit @ unit.T
    n = len(vectors)
    return float((sim.sum() - n) / (n * (n - 1)))


def clustering_attack(
    target_docs: list[Document],
    background: list[Document],
    n_mfw: int,
    rng: random.Random,
    n_control: int = 20,
) -> dict:
    combined = list(target_docs) + list(background)
    profile = MFWProfile.fit([d.text for d in combined], n_mfw)
    z_targets = np.array([profile.vectorize(d.text) for d in target_docs])
    z_background = np.array([profile.vectorize(d.text) for d in background])

    target_cohesion = _cohesion(z_targets)
    controls = [
        _cohesion(z_background[rng.sample(range(len(z_background)), len(z_targets))])
        for _ in range(n_control)
    ]
    mean_c, std_c = float(np.mean(controls)), float(np.std(controls))
    z_score = (target_cohesion - mean_c) / std_c if std_c > 1e-9 else 0.0

    authors = sorted({d.author for d in combined})
    pred = AgglomerativeClustering(
        n_clusters=len(authors), metric="cosine", linkage="average"
    ).fit_predict(np.vstack([z_targets, z_background]))
    target_labels = pred[: len(target_docs)]
    pairs = [
        (i, j)
        for i in range(len(target_docs))
        for j in range(i + 1, len(target_docs))
    ]
    same_rate = (
        float(np.mean([target_labels[i] == target_labels[j] for i, j in pairs]))
        if pairs
        else float("nan")
    )

    return {
        "n_targets": len(target_docs),
        "n_background_docs": len(background),
        "n_authors": len(authors),
        "target_cohesion": target_cohesion,
        "control_cohesion_mean": mean_c,
        "cohesion_z": float(z_score),
        "same_cluster_rate": same_rate,
        "verdict": "linkable" if z_score > 2.0 else "not distinguishable from background",
    }
