"""Attacks: clustering, cross-user null, ablation, L1 leak, detectors."""

from __future__ import annotations

import random

import numpy as np
import pytest

from harness.attacks.ablation import ablate_corpus
from harness.attacks.clustering import clustering_attack
from harness.attacks.cross_user import cross_user_attack
from harness.attacks.detectors import HeuristicAIDetector, NoopAIDetector, build_detector
from harness.attacks.l1_identification import L1LeakScorer
from harness.config import HarnessConfig
from harness.data import synthetic_corpus

CONFIG = HarnessConfig()


def test_clustering_attack_detects_linked_targets():
    rng = random.Random(CONFIG.seed)
    corpus = synthetic_corpus(n_authors=10, docs_per_author=6, words_per_doc=100, seed=CONFIG.seed)
    by_author = corpus.by_author()
    authors = sorted(by_author)
    linked = by_author[authors[0]][:6]
    background = [d for a in authors[1:] for d in by_author[a]]
    result = clustering_attack(linked, background, CONFIG.delta.n_mfw, rng)
    assert result["cohesion_z"] > 2.0
    assert result["verdict"] == "linkable"


def test_clustering_attack_null_on_independent_targets():
    rng = random.Random(CONFIG.seed + 1)
    corpus = synthetic_corpus(n_authors=10, docs_per_author=6, words_per_doc=100, seed=CONFIG.seed)
    by_author = corpus.by_author()
    authors = sorted(by_author)
    scattered = [by_author[a][0] for a in authors[:6]]
    background = [d for a in authors for d in by_author[a]]
    result = clustering_attack(scattered, background, CONFIG.delta.n_mfw, rng)
    assert result["cohesion_z"] < 2.0
    assert result["verdict"] == "not distinguishable from background"


def test_cross_user_null_holds():
    rng = random.Random(CONFIG.seed)
    corpus = synthetic_corpus(n_authors=14, docs_per_author=6, words_per_doc=100, seed=CONFIG.seed)
    result = cross_user_attack(corpus, CONFIG.delta.n_mfw, rng, k_operators=4, m_personas=3)
    assert abs(result["linking_auc_same_vs_different_operator"] - 0.5) < 0.15
    assert result["attribution_auc_sanity"] > 0.6


def test_ablation_modes():
    corpus = synthetic_corpus(n_authors=4, docs_per_author=4, words_per_doc=120, seed=3)
    fw = ablate_corpus(corpus, "function-words")
    cw = ablate_corpus(corpus, "content-words")
    assert fw.docs and cw.docs
    from harness.features import FUNCTION_WORDS

    for d in fw.docs:
        tokens = d.text.split()
        assert tokens and all(t in FUNCTION_WORDS["en"] for t in tokens)
    for d in cw.docs:
        tokens = d.text.split()
        assert tokens and all(t not in FUNCTION_WORDS["en"] for t in tokens)


def test_l1_leak_scorer_rises_on_contamination():
    rng = random.Random(CONFIG.seed)
    en = [" ".join(rng.choices(["the", "of", "and", "time", "people", "good"], k=60)) + "." for _ in range(60)]
    ru = [" ".join(rng.choices(["и", "в", "не", "что", "как", "жизнь"], k=60)) + "." for _ in range(60)]
    scorer = L1LeakScorer().fit({"en": en, "ru": ru})

    clean = scorer.accent(rng.choice(en), own_lang="en")
    mixed = rng.choice(en)[:100] + " и в не что как жизнь и в не " * 5
    accented = scorer.accent(mixed, own_lang="en")
    assert clean["top_lang"] == "en"
    assert accented["leak_prob"] > clean["leak_prob"]
    assert accented["leak_lang"] == "ru"


def test_heuristic_detector_fits_and_scores():
    rng = random.Random(1)
    human = ["Ordinary sentence here. Another one, with a comma! " * (int(rng.random() * 3) + 1) for _ in range(50)]
    det = HeuristicAIDetector().fit(human)
    s1 = det.score("Ordinary sentence here. Another one, with a comma!")
    assert s1 is not None and np.isfinite(s1)
    assert NoopAIDetector().score("text") is None
    assert build_detector("noop").name == "noop"
    with pytest.raises(KeyError):
        build_detector("nope")
