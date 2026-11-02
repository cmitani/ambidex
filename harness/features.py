"""Feature primitives: MFW z-profiles, punctuation profile, function words."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from harness.data import tokenize

_SENTENCE_SPLIT = re.compile(r"[.!?…]+")


class MFWProfile:
    """Most-frequent-word profile with z-score standardization.

    Fit on a comparison set (known + background, never the pair alone): the
    shared z-basis is what makes delta values comparable across pairs.
    """

    def __init__(self, words: list[str], mean: np.ndarray, std: np.ndarray):
        self.words = words
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, texts: list[str], n_mfw: int) -> MFWProfile:
        freqs = [Counter(tokenize(t)) for t in texts]
        total: Counter = Counter()
        for f in freqs:
            total.update(f)
        words = [w for w, _ in total.most_common(n_mfw)]
        rel = np.array(
            [[f[w] / max(1, sum(f.values())) for w in words] for f in freqs], dtype=np.float64
        )
        mean = rel.mean(axis=0)
        std = rel.std(axis=0)
        std[std < 1e-9] = 1.0
        return cls(words, mean, std)

    def vectorize(self, text: str) -> np.ndarray:
        f = Counter(tokenize(text))
        total = max(1, sum(f.values()))
        rel = np.array([f[w] / total for w in self.words], dtype=np.float64)
        return (rel - self.mean) / self.std

    def vectorize_texts_mean(self, texts: list[str]) -> np.ndarray:
        """Centroid (mean z-vector) of several docs = one author's style point."""
        if not texts:
            raise ValueError("no texts to profile")
        return np.mean([self.vectorize(t) for t in texts], axis=0)


_PUNCT_CHARS = list(".,!?;:—–-()'\"…")


def punct_vector(text: str) -> np.ndarray:
    """Punctuation & rhythm profile: rates per 1000 chars + length/caps/digit stats."""
    n = max(1, len(text))
    rates = [text.count(c) * 1000.0 / n for c in _PUNCT_CHARS]
    words = text.split()
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if sentences:
        mean_sentence = sum(len(s.split()) for s in sentences) / len(sentences)
    else:
        mean_sentence = float(len(words))
    mean_word = sum(len(w) for w in words) / max(1, len(words))
    caps = sum(1 for c in text if c.isupper()) * 1000.0 / n
    digits = sum(1 for c in text if c.isdigit()) * 1000.0 / n
    return np.array(rates + [mean_sentence, mean_word, caps, digits], dtype=np.float64)


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """Column-wise z-scoring with zero-variance guard."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-9] = 1.0
    return (matrix - mean) / std


FUNCTION_WORDS: dict[str, tuple[str, ...]] = {
    "en": ("the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "go", "see", "no", "way", "could", "people", "my", "than", "first", "been", "who", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part", "over", "new", "take", "work", "know", "place", "year", "me", "back", "give", "most", "very", "after", "thing", "our", "just", "name", "good", "where", "through", "much", "before", "too", "mean", "old", "any", "tell", "life", "boy", "want", "show", "also", "around", "form", "three", "small", "set", "put", "end", "does", "another", "well", "large", "must", "big", "even", "such", "because", "here", "why", "ask", "men", "read", "need", "land", "different", "home", "us", "move", "try", "kind", "hand", "again", "change", "off", "play", "air", "away", "house", "point", "page", "mother", "answer", "found", "study", "still", "learn", "should", "every", "near", "add", "food", "between", "own", "below", "country", "last", "school", "father", "keep", "never", "start", "city", "under", "story", "saw", "left", "few", "while", "along", "might", "close", "something", "seem", "next", "hard", "open", "example", "begin", "always", "those", "both", "together", "often", "run"),
    "ru": ("и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "сказал", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "человек", "чего", "раз", "тоже", "себе", "под", "жизнь", "будет", "ж", "тогда", "кто", "этот", "говорил", "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "кажется", "сейчас", "были", "куда", "зачем", "сказать", "всех", "никогда", "сегодня", "можно", "при", "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "сказала", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между", "очень", "просто", "ваш", "как-то", "давай"),
    "de": ("der", "die", "das", "und", "in", "den", "von", "zu", "mit", "da", "für", "auf", "ein", "eine", "des", "sich", "dem", "nicht", "ist", "ich", "es", "auch", "als", "wie", "aber", "an", "man", "so", "dass", "wir", "sie", "was", "ihm", "ihren", "diese", "einem", "sind", "werden", "aus", "hat", "kann", "gegen", "wird", "nach", "bei", "doch", "wieder", "mehr", "noch", "muss", "alles", "unter", "wurde", "sehr", "uns", "denn", "durch", "schon", "wenn", "hier", "sondern", "ohne", "euch", "ihr", "ihnen", "ob", "etwas", "meiner", "eigentlich", "mal", "nun", "einmal", "jemand", "jeder", "wenig", "gut", "groß", "klein", "lang", "kurz", "oft", "niemals", "stets", "immer", "vielleicht", "worden", "hätte", "würden", "sollten", "könnten"),
}
