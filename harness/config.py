"""Defaults for the whole harness in one place."""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class DeltaConfig:
    n_mfw: int = 300


@dataclass(frozen=True)
class CharNgramConfig:
    ngram_min: int = 3
    ngram_max: int = 5
    max_features: int = 30_000


@dataclass(frozen=True)
class UnmaskingConfig:
    n_chunks: int = 10
    min_chunk_words: int = 60
    rounds: int = 8
    eliminate: int = 3
    cv: int = 3
    max_features: int = 3_000


@dataclass(frozen=True)
class ImpostorsConfig:
    n_impostors: int = 40
    max_background_docs: int = 150


@dataclass(frozen=True)
class PoolingConfig:
    target_words: int = 400


@dataclass(frozen=True)
class PairConfig:
    eval_author_frac: float = 0.3
    neg_per_pos: int = 1


@dataclass(frozen=True)
class CalibrationConfig:
    fpr: float = 0.05
    length_bucket_bounds: tuple = (250, 1500)


@dataclass(frozen=True)
class HarnessConfig:
    seed: int = 1337
    languages: tuple = ("en", "ru", "de")

    delta: DeltaConfig = field(default_factory=DeltaConfig)
    char_ngram: CharNgramConfig = field(default_factory=CharNgramConfig)
    unmasking: UnmaskingConfig = field(default_factory=UnmaskingConfig)
    impostors: ImpostorsConfig = field(default_factory=ImpostorsConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    pairs: PairConfig = field(default_factory=PairConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def with_(self, **kwargs) -> "HarnessConfig":
        return replace(self, **kwargs)
