"""Cross-user scenario: K virtual operators x M personas, no linking allowed.

Personas are simulated by real authors (the null hypothesis: independent
writers). Persona "output" = one held-out doc, persona corpus = the rest.
Personas are grouped into fake operators at random; an adversary scoring
persona-vs-persona similarity must NOT distinguish same-operator pairs from
different-operator pairs (AUC ~ 0.5). Attribution sanity (persona output vs own
corpus must beat other corpora) proves the scores themselves work.
"""

from __future__ import annotations

import random

from sklearn.metrics import roc_auc_score

from harness.data import Corpus
from harness.features import MFWProfile
from harness.methods.deltas import cosine_distance


def cross_user_attack(
    corpus: Corpus, n_mfw: int, rng: random.Random, k_operators: int = 4, m_personas: int = 3
) -> dict:
    by_author = {a: docs for a, docs in corpus.by_author().items() if len(docs) >= 3}
    authors = sorted(by_author)
    if len(authors) < k_operators * m_personas:
        raise ValueError(f"need >= {k_operators * m_personas} authors with >=3 docs")

    chosen = rng.sample(authors, k_operators * m_personas)
    rng.shuffle(chosen)
    operator_of = {}
    personas = []
    for i, author in enumerate(chosen):
        docs = by_author[author][:]
        rng.shuffle(docs)
        operator_of[author] = i // m_personas
        personas.append((author, docs[1:], docs[0]))  # (author, corpus, output)

    texts = [d.text for _, corpus_docs, _ in personas for d in corpus_docs] + [
        out.text for _, _, out in personas
    ]
    profile = MFWProfile.fit(texts, n_mfw)
    z_outputs = [profile.vectorize(out.text) for _, _, out in personas]
    z_corpora = [profile.vectorize_texts_mean([d.text for d in docs]) for _, docs, _ in personas]

    labels, scores = [], []
    for i in range(len(personas)):
        for j in range(i + 1, len(personas)):
            same_operator = operator_of[personas[i][0]] == operator_of[personas[j][0]]
            labels.append(1 if same_operator else 0)
            scores.append(-cosine_distance(z_outputs[i], z_outputs[j]))
    linking_auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")

    attr_labels, attr_scores = [], []
    for i, (author, _, out) in enumerate(personas):
        own = -cosine_distance(z_corpora[i], z_outputs[i])
        other_idx = (i + 1) % len(personas)
        other = -cosine_distance(z_corpora[i], z_outputs[other_idx])
        attr_labels.extend([1, 0])
        attr_scores.extend([own, other])

    return {
        "k_operators": k_operators,
        "m_personas": m_personas,
        "linking_auc_same_vs_different_operator": linking_auc,
        "attribution_auc_sanity": float(roc_auc_score(attr_labels, attr_scores)),
        "verdict": (
            "null holds (no linking)"
            if abs(linking_auc - 0.5) < 0.15
            else "linking signal present — investigate"
        ),
    }
