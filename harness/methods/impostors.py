"""PAN-style impostors: share of impostor authors ranked farther than the unknown.

Rank statistic in [0, 1], robust to metric scale. NaN when the background has
< 10 other authors.
"""

from __future__ import annotations

from harness.data import nan
from harness.features import MFWProfile
from harness.methods.deltas import cosine_distance


class ImpostorsVerification:
    name = "impostors"
    description = "share of impostor authors ranked farther than the unknown (rank-based)"

    def __init__(self, name: str = "impostors"):
        self.name = name

    def score(self, known, unknown, ctx):
        cfg = ctx.config.impostors
        by_author: dict[str, list] = {}
        for d in ctx.background:
            by_author.setdefault(d.author, []).append(d)
        own = {known[0].author, unknown.author} if known else {unknown.author}
        pool = [a for a in by_author if a not in own]
        if len(pool) < 10:
            return nan()

        rng = ctx.rng
        chosen = rng.sample(pool, min(cfg.n_impostors, len(pool)))
        background_texts = [d.text for d in ctx.background[: cfg.max_background_docs]]
        profile = MFWProfile.fit(
            [d.text for d in known] + background_texts, ctx.config.delta.n_mfw
        )
        z_known = profile.vectorize_texts_mean([d.text for d in known])
        z_unknown = profile.vectorize(unknown.text)
        d_known_unknown = cosine_distance(z_known, z_unknown)

        wins = 0
        for author in chosen:
            z_imp = profile.vectorize_texts_mean([d.text for d in by_author[author]])
            if d_known_unknown < cosine_distance(z_known, z_imp):
                wins += 1
        return wins / len(chosen)
