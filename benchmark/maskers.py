"""Masker API and baselines.

A masker is a plugin: text in -> text out. Baselines are DELIBERATELY dumb —
one generic prompt, a synonym table, identity. No style targets, no persona
knowledge: anything that knows what a "persona" is does not belong here.
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse

from harness.config import HarnessConfig

SYNONYMS_EN: dict[str, tuple[str, ...]] = {
    "good": ("decent", "solid"),
    "bad": ("poor", "weak"),
    "big": ("large", "sizable"),
    "small": ("little", "modest"),
    "fast": ("quick", "rapid"),
    "slow": ("gradual", "unhurried"),
    "happy": ("glad", "cheerful"),
    "sad": ("gloomy", "downcast"),
    "smart": ("clever", "sharp"),
    "important": ("significant", "notable"),
    "difficult": ("hard", "tricky"),
    "easy": ("simple", "straightforward"),
    "strange": ("odd", "unusual"),
    "beautiful": ("lovely", "striking"),
    "angry": ("mad", "irate"),
    "tired": ("weary", "drained"),
    "start": ("begin", "kick off"),
    "end": ("finish", "wrap up"),
    "help": ("assist", "aid"),
    "make": ("build", "produce"),
    "get": ("obtain", "grab"),
    "think": ("believe", "reckon"),
    "want": ("wish", "desire"),
    "look": ("seem", "appear"),
    "say": ("mention", "note"),
    "very": ("really", "quite"),
    "really": ("truly", "genuinely"),
    "many": ("plenty of", "numerous"),
    "people": ("folks", "individuals"),
    "problem": ("issue", "trouble"),
    "answer": ("reply", "response"),
    "question": ("query", "ask"),
    "work": ("labor", "job"),
    "money": ("cash", "funds"),
    "friend": ("buddy", "companion"),
    "child": ("kid", "youngster"),
    "house": ("home", "place"),
    "day": ("afternoon", "occasion"),
    "night": ("evening", "dark hours"),
    "old": ("aged", "elderly"),
    "new": ("fresh", "recent"),
    "long": ("lengthy", "extended"),
    "short": ("brief", "quick"),
    "strong": ("sturdy", "powerful"),
    "wrong": ("mistaken", "off"),
    "right": ("correct", "accurate"),
}


class SkipMasker(Exception):
    """Raised when a masker cannot run (e.g. no local server) — not an error."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_local_service_url(url: str) -> str:
    """Allow only explicit http(s) URLs with a host and no userinfo/redirects.

    Localhost/private hosts are ALLOWED BY DESIGN: the target is the operator's
    own local inference server, configured by the operator. Everything else
    malformed (other schemes, missing host, embedded credentials) is rejected.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r} (expected http/https)")
    if not parsed.hostname:
        raise ValueError("URL has no host")
    if parsed.username or parsed.password:
        raise ValueError("userinfo in URL is not allowed")
    return url


@dataclass
class MaskerContext:
    rng: random.Random
    config: HarnessConfig
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: int = 180


class Masker(ABC):
    name: str = "masker"
    description: str = ""

    @abstractmethod
    def mask(self, text: str, ctx: MaskerContext) -> str:
        raise NotImplementedError


class IdentityMasker(Masker):
    name = "identity"
    description = "no-op: the lower bound of masking (must score WORST)"

    def mask(self, text: str, ctx: MaskerContext) -> str:
        return text


class ThesaurusMasker(Masker):
    name = "thesaurus"
    description = "random synonym substitutions at rate p (lexical-only masking)"

    def __init__(self, p: float = 0.5):
        self.p = p

    def _replace(self, match: re.Match, rng: random.Random) -> str:
        word = match.group(0)
        options = SYNONYMS_EN.get(word.lower())
        if not options or rng.random() >= self.p:
            return word
        choice = rng.choice(options)
        if word[0].isupper():
            choice = choice.capitalize()
        return choice

    def mask(self, text: str, ctx: MaskerContext) -> str:
        return re.sub(r"[A-Za-z]+", lambda m: self._replace(m, ctx.rng), text)


class OllamaMasker(Masker):
    name = "ollama"
    description = "one generic paraphrase prompt via a local server (no style target)"

    PROMPT = (
        "Paraphrase the following text, preserving its meaning and approximate "
        "length. Output only the paraphrase, nothing else.\n\n{text}"
    )

    def mask(self, text: str, ctx: MaskerContext) -> str:
        url = validate_local_service_url(ctx.ollama_url)
        payload = json.dumps(
            {
                "model": ctx.ollama_model,
                "prompt": self.PROMPT.format(text=text),
                "stream": False,
                "options": {"temperature": 0.8},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(request, timeout=ctx.ollama_timeout) as response:
                out = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            raise SkipMasker(f"server unreachable at {url}: {e}") from e
        result = str(out.get("response", "")).strip()
        if not result:
            raise SkipMasker("empty response from server")
        return result


MASKERS: dict[str, type[Masker]] = {
    IdentityMasker.name: IdentityMasker,
    ThesaurusMasker.name: ThesaurusMasker,
    OllamaMasker.name: OllamaMasker,
}


def build_masker(name: str) -> Masker:
    if name not in MASKERS:
        raise KeyError(f"unknown masker: {name}; available: {sorted(MASKERS)}")
    return MASKERS[name]()
