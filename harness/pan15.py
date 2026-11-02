"""PAN 2015/2014 author verification corpus loader.

Layout: <root>/<problem>/known*.txt + unknown.txt, <root>/truth.txt
("<problem> <Y|N>" per line). One problem = one PairSample; fit/eval split
is by problem.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from harness.data import Document
from harness.pairs import PairSample

_PROBLEM_RE = re.compile(r"^[A-Z]{2}\d+$")


def load_pan15_problems(root: str | Path, lang: str) -> list[tuple[str, list[Document], Document, int]]:
    root = Path(root)
    truth: dict[str, int] = {}
    for line in (root / "truth.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            truth[parts[0]] = 1 if parts[1].upper() == "Y" else 0
    problems = []
    for problem_dir in sorted(p for p in root.iterdir() if p.is_dir() and _PROBLEM_RE.match(p.name)):
        if problem_dir.name not in truth:
            continue
        known = [
            Document(
                doc_id=f"{problem_dir.name}/{p.name}",
                text=p.read_text(encoding="utf-8", errors="replace"),
                author=problem_dir.name,
                lang=lang,
            )
            for p in sorted(problem_dir.glob("known*.txt"))
        ]
        unknown = Document(
            doc_id=f"{problem_dir.name}/unknown",
            text=(problem_dir / "unknown.txt").read_text(encoding="utf-8", errors="replace"),
            author=f"{problem_dir.name}-unknown",
            lang=lang,
        )
        if known:
            problems.append((problem_dir.name, known, unknown, truth[problem_dir.name]))
    return problems


def pan15_to_pairs(
    problems: list[tuple[str, list[Document], Document, int]],
    eval_frac: float,
    rng: random.Random,
) -> tuple[list[PairSample], list[PairSample], list[Document]]:
    """Split problems into fit/eval pair lists; also return all known docs."""

    def make(problem) -> PairSample:
        pid, known, unknown, label = problem
        return PairSample(
            pair_id=pid,
            known_docs=known,
            unknown=unknown,
            label=label,
            lang=unknown.lang,
            known_author=pid,
            unknown_author=pid if label else None,
            exclude_from_background={pid},
        )

    ids = [p[0] for p in problems]
    rng.shuffle(ids)
    n_eval = max(1, int(len(ids) * eval_frac))
    eval_ids = set(ids[:n_eval])
    fit_pairs = [make(p) for p in problems if p[0] not in eval_ids]
    eval_pairs = [make(p) for p in problems if p[0] in eval_ids]
    all_known = [d for _, known, _, _ in problems for d in known]
    return fit_pairs, eval_pairs, all_known
