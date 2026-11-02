"""End-to-end pipeline test on a synthetic corpus (same gates as `ambidex smoke`)."""

from __future__ import annotations

import random

from harness.config import (
    CharNgramConfig,
    DeltaConfig,
    HarnessConfig,
    ImpostorsConfig,
    UnmaskingConfig,
)
from harness.data import Corpus, synthetic_corpus
from harness.pairs import build_pairs, pool_corpus
from harness.pipeline import run_experiment

CONFIG = HarnessConfig(
    delta=DeltaConfig(n_mfw=200),
    char_ngram=CharNgramConfig(max_features=10_000),
    unmasking=UnmaskingConfig(rounds=3, n_chunks=5, cv=2, eliminate=2, max_features=1200),
    impostors=ImpostorsConfig(n_impostors=15, max_background_docs=60),
)


def test_pipeline_smoke(tmp_path):
    config = CONFIG
    rng = random.Random(config.seed)
    corpus = synthetic_corpus(n_authors=14, docs_per_author=6, words_per_doc=100, seed=config.seed)
    pooled = pool_corpus(corpus, config, rng)
    fit_pairs, eval_pairs = build_pairs(pooled, config, rng)
    assert fit_pairs and eval_pairs

    by_author = pooled.by_author()
    eval_authors = {p.known_author for p in eval_pairs} | {
        p.unknown_author for p in eval_pairs if p.unknown_author
    }
    bg_fit = Corpus([d for a, docs in by_author.items() if a not in eval_authors for d in docs])
    bg_eval = Corpus([d for a, docs in by_author.items() if a in eval_authors for d in docs])

    payload = run_experiment(
        name="smoke-test",
        fit_pairs=fit_pairs,
        eval_pairs=eval_pairs,
        background_fit={"en": bg_fit.docs},
        background_eval={"en": bg_eval.docs},
        config=config,
        out_dir=tmp_path / "smoke",
        example_pair=next((p for p in eval_pairs if p.label == 1), None),
    )

    assert payload["fused"]["auc_eval"] > 0.8
    assert 0.35 < payload["shuffled_label_control_auc"] < 0.65
    assert (tmp_path / "smoke" / "report.json").exists()
    assert (tmp_path / "smoke" / "roc.png").exists()
    assert (tmp_path / "smoke" / "example_decision.txt").exists()
    assert payload["per_method"]["cosine_delta"]["auc_eval"] > 0.7
