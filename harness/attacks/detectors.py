"""AI-text detection: a SEPARATE axis from authorship, never fused into it.

score(text) -> higher = "more machine-like", or None when not applicable.
HeuristicAIDetector is a weak built-in proxy (statistical fingerprints vs the
human background); TransformersAIDetector wraps a real HF classifier and needs
the optional extras:  pip install -e ".[detectors]"
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from harness.features import punct_vector


class AIDetector(ABC):
    name: str = "detector"
    description: str = ""

    @abstractmethod
    def score(self, text: str) -> float | None:
        raise NotImplementedError


class NoopAIDetector(AIDetector):
    name = "noop"

    def score(self, text: str) -> float | None:
        return None


class HeuristicAIDetector(AIDetector):
    """Mean |z| of rhythm fingerprints vs the fitted human background (weak proxy)."""

    name = "heuristic"
    description = "statistical fingerprint vs human background (weak, zero-dep)"

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, human_texts: list[str]) -> HeuristicAIDetector:
        matrix = np.array([punct_vector(t) for t in human_texts])
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def score(self, text: str) -> float | None:
        if self.mean_ is None:
            return None
        z = (punct_vector(text) - self.mean_) / self.std_
        return float(np.mean(np.abs(z)))


class TransformersAIDetector(AIDetector):
    """HF text-classification model; requires the [detectors] extras."""

    name = "transformers"

    def __init__(self, model: str = "Hello-SimpleAI/chatgpt-detector-roberta"):
        self.model = model
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise ImportError(
                    "TransformersAIDetector needs the [detectors] extras: "
                    'pip install -e ".[detectors]"'
                ) from e
            self._pipe = pipeline("text-classification", model=self.model)
        return self._pipe

    def score(self, text: str) -> float | None:
        pipe = self._load()
        out = pipe(text[:1500], top_k=2)
        for item in out:
            label = item["label"].lower()
            if "ai" in label or "machine" in label or "fake" in label or "label_1" == label:
                return float(item["score"])
        return None


DETECTORS: dict[str, type[AIDetector]] = {
    NoopAIDetector.name: NoopAIDetector,
    HeuristicAIDetector.name: HeuristicAIDetector,
    TransformersAIDetector.name: TransformersAIDetector,
}


def build_detector(name: str) -> AIDetector:
    if name not in DETECTORS:
        raise KeyError(f"unknown detector: {name}; available: {sorted(DETECTORS)}")
    return DETECTORS[name]()
