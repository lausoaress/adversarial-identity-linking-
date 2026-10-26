"""End-to-end experiment: baseline vs. adversarial pipeline on the same
candidate pairs.

Two protocols:

  --protocol pool   (default) Pool retrieval modelled on Arabnezhad et al.:
                    each query is searched against a pool, thresholds are
                    calibrated on W1 and reported on W2, some queries have
                    no true match, and both false positives and false
                    negatives are measured. See src/evaluation/protocol.py.
  --protocol pairs  The earlier 1-vs-1 pairs check (kept for comparison).

Usage:

    python -m src.evaluation.run_experiment --provider stub --skip-llm    # baseline arms only, no model
    python -m src.evaluation.run_experiment --provider stub               # plumbing check
    python -m src.evaluation.run_experiment --provider ollama --model llama3.1 --timeout 600

This builds a synthetic alter-ego benchmark by default (see
src/data/loaders.py — make_synthetic_corpus), so it runs with zero setup.
Once you have a real corpus loaded via src/data/loaders.py, swap the
`aliases = make_synthetic_corpus(...)` line for your real loader and
everything downstream is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import sys

from src.agents.llm_client import StubLLMClient, build_client
from src.agents.orchestrator import AdversarialPipeline
from src.baseline.cosine_pipeline import BaselineConfig, accepts, run_baseline
from src.data.alter_ego import build_alter_ego_pairs, negative_pairs
from src.data.loaders import make_synthetic_corpus
from src.evaluation.metrics import adversarial_confusion, average_rounds, baseline_confusion, summarize
from src.evaluation.protocol import (
    UNCLEAR,
    PoolResult,
    bootstrap_ci,
    build_pool_task,
    compare_arms,
    count_outcomes,
    paired_sign_test,
    query_correct,
    resolution_rate,
    run_pool_protocol,
)
from src.schemas import CandidatePair, PipelineResult, Platform


def _stub_responses_for(n_pairs: int) -> list[str]:
    """Canned Proposer/Contestador/Juiz replies so `--provider stub` can
    run the whole script without any model — useful to confirm the
    plumbing works before spending time on real model calls.
    """
    one_round = [
        '{"hypothesis": "Likely the same author based on shared vocabulary.", '
        '"confidence": 0.6, "evidence": [{"description": "Shares niche vocabulary", "weight": 0.5}]}',
        '{"stance": "inconclusive", "rebuttal": "No factual contradiction found; '
        'vocabulary overlap could be coincidental given the small sample.", "evidence": []}',
        '{"verdict": "unclear", "explanation": "No contradiction, but evidence is thin.", '
        '"cited_evidence_descriptions": []}',
    ]
    return one_round * n_pairs


def _stub_responses_pool(n_queries: int) -> list[str]:
    """Same idea for the pool protocol, but the Proposer replies with a
    verdict, so the Proposer-alone arm gets something to count.
    """
    one_round = [
        '{"verdict": "unsure", "confidence": 0.5, "hypothesis": "Cannot tell.", "evidence": []}',
        '{"stance": "inconclusive", "rebuttal": "Nothing decisive either way.", "evidence": []}',
        '{"verdict": "unclear", "explanation": "Evidence is thin.", "cited_evidence_descriptions": []}',
    ]
    return one_round * n_queries


def _fmt(x: float) -> str:
    return "  n/a" if x != x else f"{x:5.2f}"


def print_pool_report(res: PoolResult) -> list[dict]:
    """Prints the W2 table and returns the same numbers as rows (for --out)."""
    w2 = res.w2
    n_match = sum(sq.matchable for sq in w2)
    print(f"\n  Thresholds: paper {res.t_paper:.4f} | recalibrated on W1 {res.t_calibrated:.4f} "
          f"| T_LOW (below this the LLM is not called) {res.t_low:.4f}")
    zones = {z: res.zones.count(z) for z in ("accept_zone", "rescue_zone", "below")}
    print(f"  W2: {len(w2)} queries ({n_match} with a true match, {len(w2) - n_match} without). "
          f"Reached the LLM: {zones['accept_zone']} in the accept zone, {zones['rescue_zone']} in the rescue zone; "
          f"{zones['below']} rejected below T_LOW.")
    print("  (a small W2 means wide intervals: read the intervals, not just the points)\n")

    header = f"  {'arm':30s} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'UNC':>3}   {'precision [95% CI]':22s} {'recall [95% CI]':22s} {'FP/query':>8} {'FN rate':>8}"
    print(header)
    rows = []
    for name, decisions in res.arms.items():
        c = count_outcomes(w2, decisions)
        p_lo, p_hi = bootstrap_ci(w2, decisions, "precision")
        r_lo, r_hi = bootstrap_ci(w2, decisions, "recall")
        print(f"  {name:30s} {c.tp:3d} {c.fp:3d} {c.fn:3d} {c.tn:3d} {c.unclear:3d}   "
              f"{_fmt(c.precision)} [{_fmt(p_lo)},{_fmt(p_hi)}]    "
              f"{_fmt(c.recall)} [{_fmt(r_lo)},{_fmt(r_hi)}]    {_fmt(c.fp_per_query):>8} {_fmt(c.fn_rate):>8}")
        rows.append({
            "arm": name, "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn, "unclear": c.unclear,
            "precision": c.precision, "precision_lo": p_lo, "precision_hi": p_hi,
            "recall": c.recall, "recall_lo": r_lo, "recall_hi": r_hi,
            "fp_per_query": c.fp_per_query, "fn_rate": c.fn_rate,
        })

    ref_name = "baseline (paper threshold)"
    ref = res.arms[ref_name]
    llm_arms = [a for a in res.arms if a in ("Proposer alone", "full pipeline")]
    if llm_arms:
        print(f"\n  Changes relative to {ref_name}:")
        ref_right = [query_correct(sq, d) for sq, d in zip(w2, ref)]
        for name in llm_arms:
            ch = compare_arms(w2, ref, res.arms[name])
            arm_right = [query_correct(sq, d) for sq, d in zip(w2, res.arms[name])]
            b_only, a_only, p = paired_sign_test(ref_right, arm_right)
            print(f"    {name:18s} FPs removed {ch['fp_removed']}, FNs rescued {ch['fn_rescued']}, "
                  f"TPs lost {ch['tp_lost']}, FPs created {ch['fp_created']} | "
                  f"right only with the LLM: {b_only}, right only with the baseline: {a_only}, sign test p = {p:.3f} | "
                  f"decisive answers in the LLM band: {_fmt(resolution_rate(res.arms[name], res.zones))}")
            rows[[r['arm'] for r in rows].index(name)].update(ch)
        parse_errors = sum(1 for r in res.llm_results.values() if r.proposer.parse_error)
        print(f"    Proposer replies that could not be parsed (after one retry): {parse_errors}/{len(res.llm_results)}")
    return rows


def _llm_from_args(args, n_calls_for_stub: int):
    llm_kwargs = {"model": args.model} if args.model else {}
    if args.provider == "ollama" and args.timeout is not None:
        llm_kwargs["timeout_s"] = args.timeout
    if args.provider == "stub":
        return StubLLMClient(scripted_responses=_stub_responses_pool(n_calls_for_stub))
    return build_client(args.provider, **llm_kwargs)


def run_pool(args) -> None:
    print(f"[1/4] Generating synthetic corpus ({args.n_users} users)...")
    aliases = make_synthetic_corpus(n_users=args.n_users, platform=Platform.OPEN_WEB)

    print("[2/4] Building the pool task (alter-ego split, no-match queries, W1/W2)...")
    task = build_pool_task(aliases, seed=args.seed, no_match_frac=args.no_match_frac, calib_frac=args.calib_frac)
    if len(task.w1) < 2 or len(task.w2) < 2:
        print("Too few users survived the minimum word/timestamp thresholds — raise --n-users.", file=sys.stderr)
        sys.exit(1)
    print(f"    pool: {len(task.pool)} known aliases | W1: {len(task.w1)} queries | W2: {len(task.w2)} queries")

    config = BaselineConfig(k=args.k)
    pipeline = None
    if not args.skip_llm:
        print(f"[3/4] Preparing the adversarial pipeline (provider={args.provider})...")
        llm = _llm_from_args(args, n_calls_for_stub=len(task.w2) + 1)
        pipeline = AdversarialPipeline.from_single_client(llm)
    else:
        print("[3/4] --skip-llm: baseline arms only.")

    print("[4/4] Scoring W1 and W2 and running the arms (the LLM only sees queries above T_LOW)...")
    res = run_pool_protocol(task, config, pipeline, target_recall=args.target_recall, low_coverage=args.low_coverage)
    rows = print_pool_report(res)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "arm", k))
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Wrote {args.out}")

    print("\nDone. This ran on SYNTHETIC data — treat these numbers as a plumbing check, not a result.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="stub", choices=["stub", "ollama", "openai", "anthropic"])
    parser.add_argument("--model", default=None, help="Model name, if the provider needs one")
    parser.add_argument("--protocol", default="pool", choices=["pool", "pairs"])
    parser.add_argument("--n-users", type=int, default=None,
                        help="Synthetic users to generate (default: 40 for pool, 12 for pairs)")
    parser.add_argument("--k", type=int, default=10, help="[pool] stage-1 shortlist size")
    parser.add_argument("--seed", type=int, default=13, help="[pool] seed for the split")
    parser.add_argument("--no-match-frac", type=float, default=0.25,
                        help="[pool] share of users whose true alias is removed from the pool")
    parser.add_argument("--calib-frac", type=float, default=0.5, help="[pool] share of queries that go to W1")
    parser.add_argument("--target-recall", type=float, default=0.80, help="[pool] recall that fixes the W1 threshold")
    parser.add_argument("--low-coverage", type=float, default=0.95,
                        help="[pool] share of correct W1 candidates that must stay above T_LOW")
    parser.add_argument("--skip-llm", action="store_true", help="[pool] run the baseline arms only")
    parser.add_argument("--out", default=None, help="[pool] write the table to this CSV file")
    parser.add_argument("--n-negatives", type=int, default=1, help="Negative pairs per positive")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Seconds to wait per Ollama call before giving up (default: OllamaClient's own, "
        "currently 300s). Raise this if you see requests.exceptions.ReadTimeout on a CPU-only "
        "machine — model load + JSON-constrained decoding can be slow on the first call.",
    )
    args = parser.parse_args()

    if args.n_users is None:
        args.n_users = 40 if args.protocol == "pool" else 12
    if args.protocol == "pool":
        run_pool(args)
        return

    print(f"[1/4] Generating synthetic corpus ({args.n_users} users)...")
    aliases = make_synthetic_corpus(n_users=args.n_users, platform=Platform.OPEN_WEB)

    print("[2/4] Building alter-ego pairs (synthetic ground truth)...")
    ae_pairs = build_alter_ego_pairs(aliases)
    neg_pairs = negative_pairs(ae_pairs, n_per_positive=args.n_negatives)
    if not ae_pairs:
        print(
            "No alter-ego pairs survived the minimum word/timestamp thresholds — "
            "raise --n-users or the synthetic corpus's messages_per_user.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = BaselineConfig()
    # Ground truth is kept PER PAIR, not per alias_id: the same "original"
    # half is reused as the dark_alias in both its one true pair and in
    # one or more negative pairs against other users' alter-egos, so an
    # alias_id -> bool dict would silently conflate them (the negative
    # entry would overwrite the positive one for the same alias_id).
    candidate_pairs: list[CandidatePair] = []
    pair_ground_truth: list[bool] = []

    for p in ae_pairs:
        candidate_pairs.append(
            CandidatePair(dark_alias=p.original, open_alias=p.alter_ego, similarity_score=None)
        )
        pair_ground_truth.append(True)
    for dark, open_ in neg_pairs:
        candidate_pairs.append(CandidatePair(dark_alias=dark, open_alias=open_, similarity_score=None))
        pair_ground_truth.append(False)

    print(f"[3/4] Running baseline cosine pipeline on {len(candidate_pairs)} pairs...")
    baseline_decisions = []
    kept_pairs: list[CandidatePair] = []
    kept_ground_truth: list[bool] = []
    for pair, same_person in zip(candidate_pairs, pair_ground_truth):
        scored = run_baseline(pair.dark_alias, [pair.open_alias], config)
        if not scored:
            continue
        scored_pair = scored[0]
        pair.similarity_score = scored_pair.similarity_score
        baseline_decisions.append((accepts(scored_pair, config), same_person))
        kept_pairs.append(pair)
        kept_ground_truth.append(same_person)

    baseline_counts = baseline_confusion(baseline_decisions)
    print("    " + summarize("Baseline", baseline_counts))

    print(f"[4/4] Running adversarial pipeline (provider={args.provider})...")
    llm_kwargs = {"model": args.model} if args.model else {}
    if args.provider == "ollama" and args.timeout is not None:
        llm_kwargs["timeout_s"] = args.timeout
    if args.provider == "stub":
        llm = StubLLMClient(scripted_responses=_stub_responses_for(len(kept_pairs)))
    else:
        llm = build_client(args.provider, **llm_kwargs)

    # Re-uses the SAME scored pairs the baseline just decided on (kept_pairs),
    # so both pipelines are compared on identical inputs, including the
    # already-computed similarity_score the Proposer logs as evidence.
    pipeline = AdversarialPipeline.from_single_client(llm)
    results: list[PipelineResult] = []
    for pair, same_person in zip(kept_pairs, kept_ground_truth):
        result = pipeline.run(pair, baseline_threshold=config.threshold)
        result.ground_truth = same_person
        results.append(result)

    adv_counts = adversarial_confusion(results)
    print("    " + summarize("Adversarial", adv_counts))
    print(f"    Average contestation rounds: {average_rounds(results):.2f}")

    print("\nDone. This ran on SYNTHETIC data — treat these numbers as a plumbing check,")
    print("not a result. Swap src/data/loaders.py's make_synthetic_corpus call for a real")
    print("corpus loader before this means anything for the TCC itself.")


if __name__ == "__main__":
    main()
