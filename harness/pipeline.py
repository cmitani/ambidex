"""End-to-end experiment runner: score -> fuse -> calibrate -> report."""

from __future__ import annotations

import math
import random
from pathlib import Path

from harness import evaluation, report
from harness.calibration import calibration_table
from harness.config import HarnessConfig
from harness.fusion import ScoreFusion
from harness.methods import build, build_all
from harness.methods.deltas import explain_top_words
from harness.pairs import PairSample


def run_experiment(
    name: str,
    fit_pairs: list[PairSample],
    eval_pairs: list[PairSample],
    background_fit: dict[str, list],
    background_eval: dict[str, list],
    config: HarnessConfig,
    out_dir: str | Path,
    method_names: list[str] | None = None,
    example_pair: PairSample | None = None,
) -> dict:
    rng = random.Random(config.seed)
    methods = build(method_names) if method_names else build_all()
    method_cols = [m.name for m in methods]

    df_fit = evaluation.score_pairs(fit_pairs, methods, background_fit, config, rng, progress=True)
    df_eval = evaluation.score_pairs(eval_pairs, methods, background_eval, config, rng, progress=True)

    fuser = ScoreFusion(method_cols).fit(df_fit)
    df_fit["fused"] = fuser.predict(df_fit)
    df_eval["fused"] = fuser.predict(df_eval)

    fpr_target = config.calibration.fpr
    per_method = {}
    for col in method_cols:
        per_method[col] = {"auc_fit": evaluation.auc(df_fit, col), "auc_eval": evaluation.auc(df_eval, col)}

    neg_eval = df_eval.loc[df_eval.label == 0, "fused"].to_numpy()
    pos_eval = df_eval.loc[df_eval.label == 1, "fused"].to_numpy()
    tau = evaluation.tau_at_fpr(neg_eval, fpr_target) if len(neg_eval) >= 5 else float("nan")
    tpr_at_tau = (
        float((pos_eval >= tau).mean()) if len(pos_eval) and not math.isnan(tau) else float("nan")
    )

    payload = {
        "experiment": name,
        "n_pairs": {"fit": len(df_fit), "eval": len(df_eval)},
        "fpr_target": fpr_target,
        "per_method": per_method,
        "fused": {
            "auc_fit": evaluation.auc(df_fit, "fused"),
            "auc_eval": evaluation.auc(df_eval, "fused"),
            "tau": tau,
            "tpr_at_tau": tpr_at_tau,
            "tpr_at_fpr_direct": evaluation.tpr_at_fpr(
                df_eval["fused"].to_numpy(), df_eval["label"].to_numpy(), fpr_target
            ),
        },
        "fused_auc_by_length_bucket": evaluation.bucket_auc(df_eval, "fused").to_dict(),
        "cosine_delta_auc_by_length_bucket": evaluation.bucket_auc(df_eval, "cosine_delta").to_dict(),
        "shuffled_label_control_auc": evaluation.shuffled_control(df_eval, "fused", rng),
        "learned_weights": fuser.weights(),
        "nan_counts_fit": fuser.nan_counts_,
        "calibration": {
            "fused": calibration_table(neg_eval, pos_eval),
            "cosine_delta": calibration_table(
                df_eval.loc[df_eval.label == 0, "cosine_delta"].dropna().to_numpy(),
                df_eval.loc[df_eval.label == 1, "cosine_delta"].dropna().to_numpy(),
            ),
        },
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.write_json(payload, out / "report.json")
    report.fig_distribution_grid(df_eval, method_cols + ["fused"], out / "distributions.png")
    report.fig_roc_curves(df_eval, method_cols + ["fused"], out / "roc.png")
    report.fig_weights(fuser.weights(), out / "weights.png")
    report.fig_calibration(df_eval, "fused", fpr_target, out / "calibration.png")

    if example_pair is not None:
        row = df_eval.loc[df_eval["pair_id"] == example_pair.pair_id]
        if len(row):
            row = row.iloc[0]
            bg = [
                d
                for d in background_eval.get(example_pair.lang, [])
                if d.author not in example_pair.exclude_from_background
            ]
            top_words = explain_top_words(
                [d.text for d in example_pair.known_docs],
                example_pair.unknown.text,
                [d.text for d in bg[: config.impostors.max_background_docs]],
                config.delta.n_mfw,
            )
            report.write_explain_report(
                out / "example_decision.txt",
                example_pair.pair_id,
                int(row["label"]),
                fuser.explain(row),
                top_words,
            )

    df_fit.to_csv(out / "scores_fit.csv", index=False)
    df_eval.to_csv(out / "scores_eval.csv", index=False)
    return payload


def print_summary(payload: dict) -> None:
    print(f"\n=== {payload['experiment']} ===")
    print(f"pairs: fit={payload['n_pairs']['fit']} eval={payload['n_pairs']['eval']}")
    print(f"{'method':<18} {'AUC fit':>8} {'AUC eval':>9}")
    for m, v in sorted(payload["per_method"].items(), key=lambda kv: -(kv[1]["auc_eval"] or 0)):
        print(f"{m:<18} {v['auc_fit']:>8.3f} {v['auc_eval']:>9.3f}")
    f = payload["fused"]
    print(f"{'[fused]':<18} {f['auc_fit']:>8.3f} {f['auc_eval']:>9.3f}")
    print(f"tau @ FPR {payload['fpr_target']:.0%}: {f['tau']:.3f}   TPR at tau: {f['tpr_at_tau']:.3f}")
    print(f"shuffled-label control AUC: {payload['shuffled_label_control_auc']:.3f} (must be ~0.5)")
    print("calibration (fused):")
    for row in payload["calibration"]["fused"]:
        print(
            f"  FPR {row['fpr']:.0%}: tau={row['tau']:.3f}  TPR@tau={row['tpr_at_tau']:.3f}"
        )
    print("learned weights:", {k: round(v, 3) for k, v in payload["learned_weights"].items()})
    print(
        "fused AUC by length bucket:",
        {k: round(v, 3) for k, v in payload["fused_auc_by_length_bucket"].items()},
    )
