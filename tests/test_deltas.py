"""Unit tests: profile math, delta correctness, chunking, pooling, pair splits."""

from __future__ import annotations

import random

import numpy as np
import pytest

from harness.config import HarnessConfig
from harness.data import Document, pool_author_documents, synthetic_corpus, tokenize
from harness.features import MFWProfile, punct_vector
from harness.methods.base import VerificationContext
from harness.methods.deltas import (
    BurrowsDelta,
    CosineDelta,
    EderDelta,
    cosine_distance,
    explain_top_words,
)
from harness.methods.unmasking import split_chunks
from harness.pairs import build_pairs


def _ctx(config=None, background=None, lang="en", seed=7):
    return VerificationContext(
        background=background or [],
        lang=lang,
        rng=random.Random(seed),
        config=config or HarnessConfig(),
    )


def _mini_corpus(seed=5, n_authors=6, docs=4, words=90):
    return synthetic_corpus(n_authors=n_authors, docs_per_author=docs, words_per_doc=words, seed=seed)


def test_mfw_profile_zero_for_basis_mean_text():
    texts = ["alpha beta gamma delta " * 10, "alpha beta gamma epsilon " * 8, "alpha zeta gamma delta " * 6]
    profile = MFWProfile.fit(texts, n_mfw=6)
    z = profile.vectorize_texts_mean(texts)
    assert np.allclose(z, 0.0, atol=1e-9)


def test_punct_vector_shape_deterministic():
    a = punct_vector("Hello, world! How: are — you? (yes) 'ok' …")
    b = punct_vector("Hello, world! How: are — you? (yes) 'ok' …")
    assert a.shape == b.shape and np.allclose(a, b)
    assert np.isfinite(a).all()


def test_burrows_delta_matches_manual_formula():
    corpus = _mini_corpus()
    by_author = corpus.by_author()
    a0, a1 = sorted(by_author)[:2]
    known = by_author[a0][:3]
    unknown = by_author[a0][3]
    background = [d for author, docs in by_author.items() if author not in (a0, a1) for d in docs]
    ctx = _ctx(background=background)

    texts = [d.text for d in known] + [d.text for d in background]
    profile = MFWProfile.fit(texts, HarnessConfig().delta.n_mfw)
    zk = profile.vectorize_texts_mean([d.text for d in known])
    zu = profile.vectorize(unknown.text)
    expected = -float(np.mean(np.abs(zk - zu)))
    assert BurrowsDelta().score(known, unknown, ctx) == pytest.approx(expected, rel=1e-9)


def test_deltas_rank_same_author_above_different():
    corpus = _mini_corpus(seed=11)
    by_author = corpus.by_author()
    a0, a1 = sorted(by_author)[:2]
    known = by_author[a0][:3]
    unknown_same = by_author[a0][3]
    unknown_other = by_author[a1][3]
    background = [d for author, docs in by_author.items() if author not in (a0, a1) for d in docs]
    ctx = _ctx(background=background)
    for method in (BurrowsDelta(), CosineDelta(), EderDelta()):
        same = method.score(known, unknown_same, ctx)
        diff = method.score(known, unknown_other, ctx)
        assert same > diff, f"{method.name}: same-author must score higher"


def test_explain_top_words_returns_profile_words():
    corpus = _mini_corpus()
    by_author = corpus.by_author()
    a0, a1 = sorted(by_author)[:2]
    words = explain_top_words(
        [d.text for d in by_author[a0][:2]],
        by_author[a0][2].text,
        [d.text for d in by_author[a1]],
        n_mfw=100,
        k=5,
    )
    assert len(words) == 5
    diffs = [d for _, d in words]
    assert diffs == sorted(diffs, reverse=True)


def test_cosine_distance_bounds():
    a = np.array([1.0, 0.0])
    assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert cosine_distance(a, np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_split_chunks_respects_minimum_words():
    text = ("Одно два три четыре пять. " * 200).strip()
    chunks = split_chunks(text, n_chunks=6, min_words=50)
    assert len(chunks) >= 3
    assert all(len(c.split()) >= 50 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == len(text.split())


def test_split_chunks_tiny_text_returns_few():
    chunks = split_chunks("Короткий текст. Еще один.", n_chunks=6, min_words=60)
    assert len(chunks) <= 2


def test_pooling_keeps_author_and_shuffles():
    docs = [Document(doc_id=f"d{i}", text=f"word{i} " * 30, author="a", lang="en") for i in range(10)]
    pooled = pool_author_documents(docs, target_words=100, rng=random.Random(1))
    assert all(p.author == "a" for p in pooled)
    assert sum(p.n_words for p in pooled) == sum(d.n_words for d in docs)


def test_build_pairs_author_disjoint_and_balanced():
    corpus = _mini_corpus(seed=3, n_authors=8, docs=5)
    config = HarnessConfig()
    rng = random.Random(config.seed)
    fit_pairs, eval_pairs = build_pairs(corpus, config, rng)
    fit_authors = {p.known_author for p in fit_pairs} | {p.unknown_author for p in fit_pairs if p.unknown_author}
    eval_authors = {p.known_author for p in eval_pairs} | {p.unknown_author for p in eval_pairs if p.unknown_author}
    assert not (fit_authors & eval_authors), "author leakage between fit and eval"
    for pairs in (fit_pairs, eval_pairs):
        labels = [p.label for p in pairs]
        assert 1 in labels and 0 in labels
        for p in pairs:
            assert p.known_author in p.exclude_from_background
            if p.unknown_author is not None:
                assert p.unknown_author in p.exclude_from_background


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World's!") == ["hello", "world's"]
