"""Benchmark: maskers, runner metrics, CLI-level integration."""

from __future__ import annotations

import random

import pytest

from benchmark.maskers import (
    IdentityMasker,
    MaskerContext,
    SkipMasker,
    ThesaurusMasker,
    build_masker,
    validate_local_service_url,
)
from benchmark.run import run_benchmark
from harness.config import HarnessConfig
from harness.data import synthetic_corpus

CONFIG = HarnessConfig()


def _ctx(seed=1):
    return MaskerContext(rng=random.Random(seed), config=CONFIG)


def test_identity_returns_text_unchanged():
    text = "Big good day, my friend."
    assert IdentityMasker().mask(text, _ctx()) == text


def test_thesaurus_changes_some_words_and_keeps_others():
    masker = ThesaurusMasker(p=1.0)
    out = masker.mask("Big good day, my friend.", _ctx())
    assert out != "Big good day, my friend."
    assert "day" not in out or "afternoon" in out  # synonym applied at p=1
    # untouched word without synonyms survives
    assert "my" in out.split() or "," in out


def test_masker_registry():
    assert build_masker("identity").name == "identity"
    with pytest.raises(KeyError):
        build_masker("nope")


def test_url_validation():
    assert validate_local_service_url("http://localhost:11434") == "http://localhost:11434"
    with pytest.raises(ValueError):
        validate_local_service_url("ftp://x")
    with pytest.raises(ValueError):
        validate_local_service_url("http://user:pass@host")


def test_ollama_masker_raises_skip_when_unreachable():
    from benchmark.maskers import OllamaMasker

    ctx = MaskerContext(rng=random.Random(1), config=CONFIG, ollama_url="http://127.0.0.1:1")
    with pytest.raises(SkipMasker):
        OllamaMasker().mask("hello world", ctx)


def test_run_benchmark_on_synthetic(tmp_path):
    corpus = synthetic_corpus(n_authors=8, docs_per_author=5, words_per_doc=100, seed=CONFIG.seed)
    by_author = corpus.by_author()
    items = [(a, docs[:-1], docs[-1]) for a, docs in sorted(by_author.items())][:6]
    payload = run_benchmark(
        items=items,
        masker_names=["identity", "thesaurus"],
        config=CONFIG,
        out_dir=tmp_path / "bench",
        progress=False,
    )
    rows = {r["masker"]: r for r in payload["rows"]}
    assert rows["identity"]["status"] == "ok"
    assert rows["thesaurus"]["status"] == "ok"
    assert rows["identity"]["style_removal_auc_cosine"] > 0.6
    assert rows["thesaurus"]["style_removal_auc_cosine"] <= rows["identity"]["style_removal_auc_cosine"] + 0.05
    assert rows["identity"]["lexical_preservation"] > 0.99
    assert (tmp_path / "bench" / "benchmark.md").exists()
    assert (tmp_path / "bench" / "report.json").exists()
