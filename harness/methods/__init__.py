"""Method registry. Add a method = subclass VerificationMethod + @register."""

from __future__ import annotations

from harness.methods.base import VerificationContext, VerificationMethod
from harness.methods.deltas import BurrowsDelta, CosineDelta, EderDelta, explain_top_words
from harness.methods.impostors import ImpostorsVerification
from harness.methods.ngrams import CharNgramSimilarity, PunctProfileSimilarity
from harness.methods.unmasking import Unmasking

REGISTRY: dict[str, type] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


register(BurrowsDelta)
register(CosineDelta)
register(EderDelta)
register(CharNgramSimilarity)
register(PunctProfileSimilarity)
register(Unmasking)
register(ImpostorsVerification)


def build_all() -> list[VerificationMethod]:
    return [cls() for cls in REGISTRY.values()]


def build(names: list[str] | None = None) -> list[VerificationMethod]:
    if not names:
        return build_all()
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise KeyError(f"unknown methods: {missing}; available: {sorted(REGISTRY)}")
    return [REGISTRY[n]() for n in names]


__all__ = [
    "REGISTRY",
    "VerificationContext",
    "VerificationMethod",
    "build",
    "build_all",
    "explain_top_words",
    "register",
]
