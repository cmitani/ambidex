"""CLI: smoke (synthetic) and verify (PAN15 dir or generic jsonl).

    python -m harness.cli smoke
    python -m harness.cli verify --pan15 data/raw/pan15/english --lang en --name pan15-en
    python -m harness.cli verify --jsonl data/interim/my_corpus.jsonl --lang ru
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from harness.config import (
    CharNgramConfig,
    DeltaConfig,
    HarnessConfig,
    ImpostorsConfig,
    UnmaskingConfig,
)
from harness.data import Corpus, Document, synthetic_corpus
from harness.pairs import build_pairs, pool_corpus
from harness.pan15 import load_pan15_problems, pan15_to_pairs
from harness.pipeline import print_summary, run_experiment

SMOKE_CONFIG = HarnessConfig(
    delta=DeltaConfig(n_mfw=250),
    char_ngram=CharNgramConfig(max_features=15_000),
    unmasking=UnmaskingConfig(rounds=4, n_chunks=6, cv=2, eliminate=2, max_features=1500),
    impostors=ImpostorsConfig(n_impostors=20, max_background_docs=80),
)


def _background_by_lang(corpus: Corpus) -> dict[str, list]:
    return {lang: corpus.filter_lang(lang).docs for lang in {d.lang for d in corpus.docs}}


def cmd_smoke(args) -> int:
    config = SMOKE_CONFIG
    rng = random.Random(config.seed)
    corpus = synthetic_corpus(n_authors=16, docs_per_author=6, words_per_doc=110, seed=config.seed)
    pooled = pool_corpus(corpus, config, rng)
    fit_pairs, eval_pairs = build_pairs(pooled, config, rng)
    print(f"synthetic corpus: {corpus.stats()}")

    by_author = pooled.by_author()
    eval_authors = {p.known_author for p in eval_pairs} | {
        p.unknown_author for p in eval_pairs if p.unknown_author
    }
    fit_bg = [d for a, docs in by_author.items() if a not in eval_authors for d in docs]
    eval_bg = [d for a, docs in by_author.items() if a in eval_authors for d in docs]

    payload = run_experiment(
        name="smoke",
        fit_pairs=fit_pairs,
        eval_pairs=eval_pairs,
        background_fit=_background_by_lang(Corpus(fit_bg)),
        background_eval=_background_by_lang(Corpus(eval_bg)),
        config=config,
        out_dir=Path(args.out) / "smoke",
        example_pair=next((p for p in eval_pairs if p.label == 1), None),
    )
    print_summary(payload)

    fused_auc = payload["fused"]["auc_eval"]
    control = payload["shuffled_label_control_auc"]
    ok = fused_auc > 0.8 and abs(control - 0.5) < 0.15
    print(
        f"\nsmoke gate: fused eval AUC {fused_auc:.3f} (>0.8), control {control:.3f} (~0.5) -> "
        + ("PASS" if ok else "FAIL")
    )
    return 0 if ok else 1


def _load_jsonl(path: str, lang: str) -> Corpus:
    docs = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            docs.append(
                Document(
                    doc_id=str(row.get("doc_id", f"row{i}")),
                    text=row["text"],
                    author=str(row["author"]),
                    lang=row.get("lang", lang),
                )
            )
    return Corpus(docs)


def cmd_verify(args) -> int:
    config = HarnessConfig()
    rng = random.Random(config.seed)
    if args.pan15:
        problems = load_pan15_problems(args.pan15, args.lang)
        if not problems:
            print(f"no problems found under {args.pan15}", file=sys.stderr)
            return 2
        labels = [p[3] for p in problems]
        print(f"PAN corpus: {len(problems)} problems ({sum(labels)} same / {len(labels) - sum(labels)} diff)")
        fit_pairs, eval_pairs, all_known = pan15_to_pairs(problems, args.eval_frac, rng)
        fit_ids = {p.pair_id for p in fit_pairs}
        eval_ids = {p.pair_id for p in eval_pairs}
        bg_fit = [d for d in all_known if d.author.split("/")[0] in fit_ids]
        bg_eval = [d for d in all_known if d.author.split("/")[0] in eval_ids]
        background_fit = {args.lang: bg_fit}
        background_eval = {args.lang: bg_eval}
        example = next((p for p in eval_pairs if p.label == 1), None)
    elif args.jsonl:
        corpus = _load_jsonl(args.jsonl, args.lang).filter_lang(args.lang)
        pooled = pool_corpus(corpus, config, rng)
        fit_pairs, eval_pairs = build_pairs(pooled, config, rng)
        by_author = pooled.by_author()
        eval_authors = {p.known_author for p in eval_pairs} | {
            p.unknown_author for p in eval_pairs if p.unknown_author
        }
        bg_fit = [d for a, docs in by_author.items() if a not in eval_authors for d in docs]
        bg_eval = [d for a, docs in by_author.items() if a in eval_authors for d in docs]
        background_fit = {args.lang: bg_fit}
        background_eval = {args.lang: bg_eval}
        example = next((p for p in eval_pairs if p.label == 1), None)
        print(f"corpus: {corpus.stats()}")
    else:
        print("specify --pan15 or --jsonl", file=sys.stderr)
        return 2

    payload = run_experiment(
        name=args.name,
        fit_pairs=fit_pairs,
        eval_pairs=eval_pairs,
        background_fit=background_fit,
        background_eval=background_eval,
        config=config,
        out_dir=Path(args.out) / args.name,
        method_names=args.methods.split(",") if args.methods else None,
        example_pair=example,
    )
    print_summary(payload)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ambidex")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="end-to-end run on a synthetic corpus")
    p_smoke.add_argument("--out", default="reports")
    p_smoke.set_defaults(func=cmd_smoke)

    p_verify = sub.add_parser("verify", help="score a real corpus")
    p_verify.add_argument("--pan15", help="extracted PAN15 corpus dir (problem dirs + truth.txt)")
    p_verify.add_argument("--jsonl", help="generic corpus: jsonl rows {text, author, lang?}")
    p_verify.add_argument("--lang", default="en")
    p_verify.add_argument("--name", default="verify")
    p_verify.add_argument("--out", default="reports")
    p_verify.add_argument("--eval-frac", type=float, default=0.3)
    p_verify.add_argument("--methods", default=None, help="comma-separated subset of methods")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
