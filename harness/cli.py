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

from harness.attacks.ablation import ablate_corpus
from harness.config import (
    CharNgramConfig,
    DeltaConfig,
    HarnessConfig,
    ImpostorsConfig,
    UnmaskingConfig,
)
from harness.data import Corpus, Document, synthetic_corpus
from harness.features import FUNCTION_WORDS
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
        problems = _ablate_problems(problems, args.ablate)
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
        if args.ablate != "none":
            corpus = ablate_corpus(corpus, args.ablate)
            print(f"ablation '{args.ablate}': {corpus.stats()}")
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


def _ablate_problems(problems, mode: str):
    if mode == "none":
        return problems
    out = []
    for pid, known, unknown, label in problems:
        ablated = ablate_corpus(Corpus(known + [unknown]), mode).docs
        known_abl = [d for d in ablated if not d.doc_id.endswith("/unknown")]
        unknown_abl = [d for d in ablated if d.doc_id.endswith("/unknown")]
        if known_abl and unknown_abl:
            out.append((pid, known_abl, unknown_abl[0], label))
    print(f"ablation '{mode}': {len(out)}/{len(problems)} problems survived")
    return out


def cmd_attack(args) -> int:
    from harness.attacks.clustering import clustering_attack
    from harness.attacks.cross_user import cross_user_attack
    from harness.attacks.l1_identification import L1LeakScorer

    config = HarnessConfig()
    rng = random.Random(config.seed)
    args.attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    if args.synthetic:
        corpus = synthetic_corpus(n_authors=12, docs_per_author=6, words_per_doc=110, seed=config.seed)
    elif args.pan15:
        problems = load_pan15_problems(args.pan15, args.lang)
        docs = [d for _, known, _, _ in problems for d in known]
        corpus = Corpus(docs)
    elif args.jsonl:
        corpus = _load_jsonl(args.jsonl, args.lang).filter_lang(args.lang)
    else:
        print("specify --synthetic, --pan15 or --jsonl", file=sys.stderr)
        return 2
    print(f"corpus: {corpus.stats()}")
    results: dict = {"corpus": corpus.stats()}

    by_author = corpus.by_author()
    prolific = sorted(by_author, key=lambda a: -len(by_author[a]))

    if "clustering" in args.attacks:
        if len(prolific) < 5:
            results["clustering"] = "skipped: need >= 5 authors"
        else:
            target_authors = prolific[:3]
            targets = [d for a in target_authors for d in by_author[a]][:18]
            rest = [d for a in prolific[3:] for d in by_author[a]]
            background = rng.sample(rest, min(len(rest), 120))
            results["clustering"] = clustering_attack(
                targets, background, config.delta.n_mfw, rng
            )

    if "cross-user" in args.attacks:
        try:
            results["cross_user"] = cross_user_attack(
                corpus, config.delta.n_mfw, rng, k_operators=args.k, m_personas=args.m
            )
        except ValueError as e:
            results["cross_user"] = f"skipped: {e}"

    if "l1" in args.attacks:
        lang_texts: dict[str, list[str]] = {}
        for lang in {d.lang for d in corpus.docs}:
            texts = [d.text for d in corpus.filter_lang(lang).docs]
            if len(texts) >= 10:
                lang_texts[lang] = texts
        if args.synthetic:
            lang_texts.setdefault(
                "ru",
                [
                    " ".join(rng.choices(list(FUNCTION_WORDS["ru"]), k=70)) + "."
                    for _ in range(40)
                ],
            )
        if len(lang_texts) < 2:
            results["l1"] = "skipped: need >= 2 languages with >= 10 docs each"
        else:
            scorer = L1LeakScorer().fit(lang_texts)
            sample_lang = max(lang_texts, key=lambda l: len(lang_texts[l]))
            clean = scorer.accent(rng.choice(lang_texts[sample_lang]), own_lang=sample_lang)
            foreign = next(l for l in lang_texts if l != sample_lang)
            foreign_vocab = " ".join(lang_texts[foreign]).split()[:2000]
            mixed_text = (
                " ".join(rng.choice(lang_texts[sample_lang]).split()[:40])
                + " "
                + " ".join(rng.choices(foreign_vocab, k=40))
            )
            accented = scorer.accent(mixed_text, own_lang=sample_lang)
            results["l1"] = {
                "clean_text": clean,
                "accented_text": accented,
                "leak_prob_delta": accented["leak_prob"] - clean["leak_prob"],
            }

    out = Path(args.out) / f"attacks-{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    return 0


def cmd_bench(args) -> int:
    from benchmark.run import run_benchmark

    config = HarnessConfig()
    rng = random.Random(config.seed)
    if args.synthetic:
        corpus = synthetic_corpus(n_authors=12, docs_per_author=6, words_per_doc=110, seed=config.seed)
        by_author = corpus.by_author()
        items = [
            (a, docs[:-1], docs[-1]) for a, docs in sorted(by_author.items())
        ][: args.n]
    elif args.pan15:
        problems = load_pan15_problems(args.pan15, args.lang)
        # style-removal metrics need true same-author items: filter to label Y
        problems = [p for p in problems if p[3] == 1]
        problems = rng.sample(problems, min(args.n, len(problems)))
        items = [(pid, known, unknown) for pid, known, unknown, _ in problems]
    elif args.jsonl:
        corpus = pool_corpus(
            _load_jsonl(args.jsonl, args.lang).filter_lang(args.lang), config, rng
        )
        by_author = corpus.by_author()
        items = [
            (a, docs[:-1], docs[-1])
            for a, docs in sorted(by_author.items())
            if len(docs) >= 2
        ][: args.n]
    else:
        print("specify --synthetic, --pan15 or --jsonl", file=sys.stderr)
        return 2

    maskers = [m.strip() for m in args.maskers.split(",") if m.strip()]
    print(f"benchmark: {len(items)} items, maskers: {maskers}")
    payload = run_benchmark(
        items=items,
        masker_names=maskers,
        config=config,
        out_dir=Path(args.out) / f"bench-{args.name}",
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )
    for row in payload["rows"]:
        print(row)
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
    p_verify.add_argument(
        "--ablate",
        default="none",
        choices=["none", "function-words", "content-words"],
        help="content ablation applied to the corpus before pairing",
    )
    p_verify.set_defaults(func=cmd_verify)

    p_attack = sub.add_parser("attack", help="run standalone attacks on a corpus")
    p_attack.add_argument("--synthetic", action="store_true", help="use a synthetic corpus")
    p_attack.add_argument("--pan15", help="extracted PAN15 corpus dir")
    p_attack.add_argument("--jsonl", help="generic corpus: jsonl rows {text, author, lang?}")
    p_attack.add_argument("--lang", default="en")
    p_attack.add_argument("--name", default="run")
    p_attack.add_argument("--out", default="reports")
    p_attack.add_argument(
        "--attacks", default="clustering,cross-user,l1", help="comma-separated attack list"
    )
    p_attack.add_argument("--k", type=int, default=4, help="cross-user: virtual operators")
    p_attack.add_argument("--m", type=int, default=3, help="cross-user: personas per operator")
    p_attack.set_defaults(func=cmd_attack)

    p_bench = sub.add_parser("bench", help="run baseline maskers through the attack battery")
    p_bench.add_argument("--synthetic", action="store_true", help="use a synthetic corpus")
    p_bench.add_argument("--pan15", help="extracted PAN15 corpus dir")
    p_bench.add_argument("--jsonl", help="generic corpus: jsonl rows {text, author, lang?}")
    p_bench.add_argument("--lang", default="en")
    p_bench.add_argument("--name", default="run")
    p_bench.add_argument("--out", default="reports")
    p_bench.add_argument("--n", type=int, default=20, help="number of items to mask")
    p_bench.add_argument("--maskers", default="identity,thesaurus", help="comma-separated masker list")
    p_bench.add_argument("--ollama-url", default="http://localhost:11434")
    p_bench.add_argument("--ollama-model", default="qwen2.5:7b")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
