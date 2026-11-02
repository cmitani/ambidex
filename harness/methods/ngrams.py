"""Surface channels: char n-grams and punctuation profile (decorrelated from Deltas)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from harness.features import punct_vector, zscore_columns
from harness.methods.base import VerificationMethod


class CharNgramSimilarity(VerificationMethod):
    name = "char_ngram"
    description = "TF-IDF char n-gram cosine (punctuation/morphology/idiosyncrasies)"

    def score(self, known, unknown, ctx):
        cfg = ctx.config.char_ngram
        texts = [d.text for d in known] + [d.text for d in ctx.background]
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            max_features=cfg.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        vec.fit(texts)
        known_matrix = vec.transform([d.text for d in known])
        known_profile = np.asarray(known_matrix.mean(axis=0))
        unknown_vec = vec.transform([unknown.text])
        return float(cosine_similarity(known_profile, unknown_vec)[0, 0])


class PunctProfileSimilarity(VerificationMethod):
    name = "punct_profile"
    description = "cosine over z-scored punctuation/rhythm vectors (language-agnostic)"

    def score(self, known, unknown, ctx):
        known_vecs = np.array([punct_vector(d.text) for d in known])
        background_vecs = np.array([punct_vector(d.text) for d in ctx.background])
        basis = np.vstack([background_vecs, known_vecs]) if len(background_vecs) else known_vecs
        z_basis = zscore_columns(basis)
        n_bg = len(background_vecs)
        z_known = z_basis[n_bg:].mean(axis=0)
        std, mean = basis.std(axis=0), basis.mean(axis=0)
        std[std < 1e-9] = 1.0
        z_unknown = (punct_vector(unknown.text) - mean) / std
        a, b = z_known - z_known.mean(), z_unknown - z_unknown.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
