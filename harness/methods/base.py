"""Verification method interface.

Contract:
- score is HIGHER => "same author" more likely; distances return negated values.
- NaN means "not applicable for this input", never evidence.
- Stateless between calls; per-pair fitting on known+background only.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from harness.config import HarnessConfig
from harness.data import Document


@dataclass
class VerificationContext:
    """Per-pair context; background must not contain the pair's own authors."""

    background: list[Document]
    lang: str
    rng: random.Random
    config: HarnessConfig


class VerificationMethod(ABC):
    name: str = "method"
    description: str = ""

    @abstractmethod
    def score(self, known: list[Document], unknown: Document, ctx: VerificationContext) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"
