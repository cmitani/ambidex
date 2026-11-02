"""Content ablation: function-words-only vs content-words-only corpora.

Answers "how much of the verifier's signal is TOPIC, not STYLE": if AUC on
function-words-only barely drops, the signal is style; if it collapses, the
verifier was reading the topic.
"""

from __future__ import annotations

from harness.data import Corpus, Document, tokenize
from harness.features import FUNCTION_WORDS


def ablate_corpus(corpus: Corpus, mode: str) -> Corpus:
    """mode: 'function-words' keeps only function words, 'content-words' drops them."""
    if mode not in ("function-words", "content-words"):
        raise ValueError(f"unknown ablation mode: {mode}")
    out: list[Document] = []
    for d in corpus.docs:
        keep = FUNCTION_WORDS.get(d.lang, ())
        tokens = tokenize(d.text)
        if mode == "function-words":
            kept = [t for t in tokens if t in keep]
        else:
            kept = [t for t in tokens if t not in keep]
        if len(kept) >= 10:
            out.append(Document(d.doc_id, " ".join(kept), d.author, d.lang, d.timestamp))
    return Corpus(out)
