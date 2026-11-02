"""L1-leak probe: does a text in language A carry readable traces of language B?

A multiclass char-n-gram model is trained on monolingual corpora per language;
for any text it returns P(lang). For a clean L2 text the second-language share
stays near its corpus baseline; a text "with an accent" (L1 function words,
punctuation habits, calques leaking through) shifts probability mass toward the
L1. This is the infrastructure + a weak proxy: a proper L1-id attack needs
non-native-speaker corpora, which is a documented open question.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class L1LeakScorer:
    def __init__(self, ngram_range=(2, 4), max_features=40_000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.langs_: list[str] = []
        self.vec_: TfidfVectorizer | None = None
        self.clf_: LogisticRegression | None = None

    def fit(self, lang_texts: dict[str, list[str]]) -> L1LeakScorer:
        texts, labels = [], []
        for lang, docs in lang_texts.items():
            texts.extend(docs)
            labels.extend([lang] * len(docs))
        self.langs_ = sorted(lang_texts)
        self.vec_ = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        X = self.vec_.fit_transform(texts)
        self.clf_ = LogisticRegression(max_iter=1000)
        self.clf_.fit(X, labels)
        return self

    def accent(self, text: str, own_lang: str | None = None) -> dict:
        """P(lang) distribution; with own_lang given, leak = strongest FOREIGN language."""
        if self.clf_ is None:
            raise RuntimeError("fit() first")
        proba = self.clf_.predict_proba(self.vec_.transform([text]))[0]
        classes = list(self.clf_.classes_)
        top_i = int(np.argmax(proba))
        if own_lang is not None and own_lang in classes:
            foreign = [(p, c) for p, c in zip(proba, classes) if c != own_lang]
            leak_prob, leak_lang = max(foreign)
        else:
            order = np.argsort(proba)[::-1]
            leak_lang = classes[order[1]] if len(order) > 1 else classes[order[0]]
            leak_prob = float(proba[order[1]])
        return {
            "top_lang": str(classes[top_i]),
            "top_prob": float(proba[top_i]),
            "leak_lang": str(leak_lang),
            "leak_prob": float(leak_prob),
        }
