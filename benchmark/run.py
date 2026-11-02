"""Benchmark runner: masker x attack -> metric table.

For every masker:
- style_removal_auc: verification AUC of (author corpus vs own masked text)
  against (author corpus vs someone else's masked text). 0.5 = style destroyed
  (perfect masking), 1.0 = untouched style (no masking). Identity must sit near
  the top, a real masker must push it toward 0.5.
- lexical_preservation: TF-IDF cosine between original and masked text (a
  meaning PROXY — embeddings-based semantic score is a later upgrade).
- cohesion_delta: mean pairwise similarity among masked outputs minus among
  originals. Positive = the outputs started to look like one family (bad).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from benchmark.maskers import MaskerContext, SkipMasker, build_masker
from harness.attacks.clustering import _cohesion
from harness.config import HarnessConfig
from harness.data import Document
from harness.features import MFWProfile
from harness.methods.base import VerificationContext
from harness.methods.deltas import BurrowsDelta, CosineDelta


def run_benchmark(
    items: list[tuple[str, list[Document], Document]],
    masker_names: list[str],
    config: HarnessConfig,
    out_dir: str | Path,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:7b",
    progress: bool = True,
) -> dict:
    """items: (author_id, known docs, unknown doc). Writes report.json + benchmark.md."""
    rng = random.Random(config.seed)
    ctx = MaskerContext(rng=rng, config=config, ollama_url=ollama_url, ollama_model=ollama_model)
    all_known = [d for _, known, _ in items for d in known]
    originals = [u.text for _, _, u in items]

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=20_000, dtype=np.float32)
    vec.fit(originals + [d.text for d in all_known])
    original_vecs = vec.transform(originals)
    original_cohesion = _cohesion(
        np.array([MFWProfile.fit(originals, config.delta.n_mfw).vectorize(t) for t in originals])
    )

    rows: list[dict] = []
    for masker_name in masker_names:
        masker = build_masker(masker_name)
        row: dict = {"masker": masker_name, "description": masker.description}
        try:
            masked_texts = []
            for i, (_, _, unknown) in enumerate(items):
                masked_texts.append(masker.mask(unknown.text, ctx))
                if progress and (i + 1) % 10 == 0:
                    print(f"  [{masker_name}] masked {i + 1}/{len(items)}")
        except SkipMasker as e:
            row["status"] = f"skipped: {e}"
            rows.append(row)
            continue
        row["status"] = "ok"

        masked_docs = [
            Document(f"masked-{i}", t, items[i][0], items[i][2].lang)
            for i, t in enumerate(masked_texts)
        ]

        pos_scores, neg_scores = [], []
        pos_scores_b, neg_scores_b = [], []
        for i, (author, known, _) in enumerate(items):
            background = [d for d in all_known if d.author != author]
            ctx_eval = VerificationContext(
                background=background, lang=masked_docs[i].lang, rng=rng, config=config
            )
            j = (i + 1) % len(items)
            pos_scores.append(CosineDelta().score(known, masked_docs[i], ctx_eval))
            neg_scores.append(CosineDelta().score(known, masked_docs[j], ctx_eval))
            pos_scores_b.append(BurrowsDelta().score(known, masked_docs[i], ctx_eval))
            neg_scores_b.append(BurrowsDelta().score(known, masked_docs[j], ctx_eval))

        row["style_removal_auc_cosine"] = _auc(pos_scores, neg_scores)
        row["style_removal_auc_burrows"] = _auc(pos_scores_b, neg_scores_b)

        masked_vecs = vec.transform(masked_texts)
        row["lexical_preservation"] = float(
            np.mean(cosine_similarity(original_vecs, masked_vecs).diagonal())
        )

        profile = MFWProfile.fit(masked_texts + [d.text for d in all_known], config.delta.n_mfw)
        masked_cohesion = _cohesion(
            np.array([profile.vectorize(t) for t in masked_texts])
        )
        row["output_cohesion_delta"] = float(masked_cohesion - original_cohesion)
        rows.append(row)

    payload = {"n_items": len(items), "rows": rows}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import json

    (out / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    (out / "benchmark.md").write_text(_markdown_table(rows), encoding="utf-8")
    return payload


def _auc(pos: list[float], neg: list[float]) -> float:
    labels = [1] * len(pos) + [0] * len(neg)
    scores = pos + neg
    if len(set(labels)) < 2 or any(np.isnan(scores)):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _markdown_table(rows: list[dict]) -> str:
    headers = [
        "masker",
        "status",
        "style_removal_auc_cosine",
        "style_removal_auc_burrows",
        "lexical_preservation",
        "output_cohesion_delta",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    note = (
        "\n\nstyle_removal_auc: 1.0 = no masking effect, 0.5 = style destroyed "
        "(goal of a masker). lexical_preservation: TF-IDF proxy of meaning kept. "
        "output_cohesion_delta > 0 = masked outputs drifted toward one family look.\n"
    )
    return "\n".join(lines) + note
