"""Pair construction with author-disjoint fit/eval splits.

Positive: known = author's docs except one, unknown = held-out doc, label 1.
Negative: known = author A's docs, unknown = a doc of another real author B.
An author never appears in both splits — otherwise the fusion memorizes authors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from harness.config import HarnessConfig
from harness.data import Corpus, Document, pool_author_documents


@dataclass
class PairSample:
    pair_id: str
    known_docs: list[Document]
    unknown: Document
    label: int  # 1 = same author, 0 = different
    lang: str
    known_author: str
    unknown_author: str | None = None
    exclude_from_background: set[str] = field(default_factory=set)


def build_pairs(
    corpus: Corpus, config: HarnessConfig, rng: random.Random
) -> tuple[list[PairSample], list[PairSample]]:
    """Return (fit_pairs, eval_pairs) with disjoint author sets per language."""
    cfg = config.pairs
    fit_pairs: list[PairSample] = []
    eval_pairs: list[PairSample] = []
    for lang in {d.lang for d in corpus.docs}:
        by_author = corpus.filter_lang(lang).by_author()
        eligible = {a: docs for a, docs in by_author.items() if len(docs) >= 2}
        authors = sorted(eligible)
        rng.shuffle(authors)
        n_eval = max(1, int(len(authors) * cfg.eval_author_frac)) if len(authors) > 3 else 0
        eval_set = set(authors[:n_eval])
        fit_set = set(authors) - eval_set
        splits = ((fit_set, fit_pairs), (eval_set, eval_pairs))
        for split_authors, split_pairs in splits:
            for author in sorted(split_authors):
                docs = eligible[author][:]
                rng.shuffle(docs)
                known, unknown = docs[:-1], docs[-1]
                split_pairs.append(
                    PairSample(
                        pair_id=f"{author}|pos",
                        known_docs=known,
                        unknown=unknown,
                        label=1,
                        lang=lang,
                        known_author=author,
                        unknown_author=author,
                        exclude_from_background={author},
                    )
                )
                others = [a for a in sorted(split_authors) if a != author]
                for j, other in enumerate(rng.sample(others, min(cfg.neg_per_pos, len(others)))):
                    other_docs = eligible[other][:]
                    neg_unknown = rng.choice(other_docs)
                    split_pairs.append(
                        PairSample(
                            pair_id=f"{author}|neg{j}",
                            known_docs=known,
                            unknown=neg_unknown,
                            label=0,
                            lang=lang,
                            known_author=author,
                            unknown_author=other,
                            exclude_from_background={author, other},
                        )
                    )
    return fit_pairs, eval_pairs


def pool_corpus(corpus: Corpus, config: HarnessConfig, rng: random.Random) -> Corpus:
    """Pool short documents per author before pair construction."""
    pooled: list[Document] = []
    for docs in corpus.by_author().values():
        pooled.extend(pool_author_documents(docs, config.pooling.target_words, rng))
    return Corpus(pooled)
