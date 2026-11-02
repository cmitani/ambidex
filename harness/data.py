"""Documents, corpora, tokenization, pooling, synthetic corpus generation."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Unicode word tokenizer, lowercased; feeds word-frequency channels only."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class Document:
    doc_id: str
    text: str
    author: str
    lang: str
    timestamp: str | None = None

    @property
    def n_words(self) -> int:
        return len(tokenize(self.text))

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass
class Corpus:
    docs: list[Document] = field(default_factory=list)

    def filter_lang(self, lang: str) -> Corpus:
        return Corpus([d for d in self.docs if d.lang == lang])

    def by_author(self) -> dict[str, list[Document]]:
        out: dict[str, list[Document]] = defaultdict(list)
        for d in self.docs:
            out[d.author].append(d)
        return dict(out)

    def authors(self) -> list[str]:
        return sorted(self.by_author())

    def stats(self) -> dict:
        per_author = self.by_author()
        words = [d.n_words for d in self.docs]
        return {
            "documents": len(self.docs),
            "authors": len(per_author),
            "langs": sorted({d.lang for d in self.docs}),
            "words_total": sum(words),
            "words_median": sorted(words)[len(words) // 2] if words else 0,
        }


def pool_author_documents(
    docs: list[Document], target_words: int, rng: random.Random
) -> list[Document]:
    """Pool short docs into ~target_words pseudo-docs; shuffled order, docs never split."""
    if not docs:
        return []
    order = list(docs)
    rng.shuffle(order)
    pools: list[list[Document]] = []
    current: list[Document] = []
    current_words = 0
    for doc in order:
        w = doc.n_words
        if current and current_words + w > max(target_words, w):
            pools.append(current)
            current, current_words = [], 0
        current.append(doc)
        current_words += w
        if current_words >= target_words:
            pools.append(current)
            current, current_words = [], 0
    if current:
        pools.append(current)
    out = []
    for i, pool in enumerate(pools):
        text = "\n\n".join(d.text for d in pool)
        out.append(
            Document(
                doc_id=f"{pool[0].doc_id}#pool{i}",
                text=text,
                author=docs[0].author,
                lang=docs[0].lang,
                timestamp=pool[-1].timestamp,
            )
        )
    return out


_COMMON_WORDS_EN = ["the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "go", "see", "number", "no", "way", "could", "people", "my", "than", "first", "water", "been", "call", "who", "oil", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part", "over", "new", "sound", "take", "work", "know", "place", "year", "live", "me", "back", "give", "most", "very", "after", "thing", "our", "just", "name", "good", "sentence", "man", "think", "say", "great", "where", "help", "through", "much", "before", "line", "right", "too", "mean", "old", "any", "tell", "life", "following", "boy", "come", "want", "show", "also", "around", "form", "three", "small", "set", "put", "end", "does", "another", "well", "large", "must", "big", "even", "such", "because", "turn", "here", "why", "ask", "went", "men", "read", "need", "land", "different", "home", "us", "move", "try", "kind", "hand", "picture", "again", "change", "off", "play", "spell", "air", "away", "animal", "house", "point", "page", "letter", "mother", "answer", "found", "study", "still", "learn", "should", "america", "world", "high", "every", "near", "add", "food", "between", "own", "below", "country", "plant", "last", "school", "father", "keep", "tree", "never", "start", "city", "earth", "eyes", "light", "thought", "head", "under", "story", "saw", "left", "dont", "few", "while", "along", "might", "close", "something", "seem", "next", "hard", "open", "example", "begin", "life", "always", "those", "both", "paper", "together", "got", "group", "often", "run"]


@dataclass
class _StyleGenome:
    vocab_weights: list[float]
    sentence_mu: float
    sentence_sigma: float
    comma_rate: float
    exclaim_rate: float
    question_rate: float


def _make_genome(rng: random.Random, vocab_size: int) -> _StyleGenome:
    vocab = _COMMON_WORDS_EN[:vocab_size]
    weights = [rng.random() + 0.05 for _ in vocab]
    total = sum(weights)
    return _StyleGenome(
        vocab_weights=[w / total for w in weights],
        sentence_mu=rng.uniform(5.0, 16.0),
        sentence_sigma=rng.uniform(1.5, 4.0),
        comma_rate=rng.uniform(0.02, 0.25),
        exclaim_rate=rng.uniform(0.0, 0.12),
        question_rate=rng.uniform(0.0, 0.10),
    )


def _generate_doc(genome: _StyleGenome, n_words: int, rng: random.Random) -> str:
    vocab = _COMMON_WORDS_EN[: len(genome.vocab_weights)]
    sentences = []
    remaining = n_words
    while remaining > 0:
        length = max(3, min(remaining, int(rng.gauss(genome.sentence_mu, genome.sentence_sigma))))
        words = rng.choices(vocab, weights=genome.vocab_weights, k=length)
        sentence = " ".join(words)
        if rng.random() < genome.comma_rate:
            cut = max(2, length // 2)
            sentence = " ".join(words[:cut]) + ", " + " ".join(words[cut:])
        r = rng.random()
        if r < genome.exclaim_rate:
            sentence += "!"
        elif r < genome.exclaim_rate + genome.question_rate:
            sentence += "?"
        else:
            sentence += "."
        sentences.append(sentence)
        remaining -= length
    return " ".join(sentences)


def synthetic_corpus(
    n_authors: int = 12,
    docs_per_author: int = 6,
    words_per_doc: int = 110,
    lang: str = "en",
    seed: int = 1337,
) -> Corpus:
    """Test fixture: per-author style genomes, deliberately easy for the methods."""
    rng = random.Random(seed)
    genomes = {a: _make_genome(rng, len(_COMMON_WORDS_EN)) for a in range(n_authors)}
    docs = []
    for author in range(n_authors):
        for j in range(docs_per_author):
            words = max(40, int(rng.gauss(words_per_doc, 25)))
            docs.append(
                Document(
                    doc_id=f"syn-a{author}-d{j}",
                    text=_generate_doc(genomes[author], words, rng),
                    author=f"syn-a{author}",
                    lang=lang,
                )
            )
    return Corpus(docs)


def nan() -> float:
    """'method not applicable' marker."""
    return math.nan
