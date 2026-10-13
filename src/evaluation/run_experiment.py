"""End-to-end experiment: baseline vs. adversarial pipeline on the same
candidate pairs.

Usage:

    python -m src.evaluation.run_experiment --provider ollama --model llama3.1
    python -m src.evaluation.run_experiment --provider stub   # no model needed, sanity check only

This builds a synthetic alter-ego benchmark by default (see
src/data/loaders.py — make_synthetic_corpus), so it runs with zero setup.
Once you have a real corpus loaded via src/data/loaders.py, swap the
`aliases = make_synthetic_corpus(...)` line for your real loader and
everything downstream is unchanged.
"""
from __future__ import annotations

import argparse
import sys

from src.agents.llm_client import StubLLMClient, build_client
from src.agents.orchestrator import AdversarialPipeline
from src.baseline.cosine_pipeline import BaselineConfig, accepts, run_baseline
from src.data.alter_ego import build_alter_ego_pairs, negative_pairs
from src.data.loaders import make_synthetic_corpus
from src.evaluation.metrics import adversarial_confusion, average_rounds, baseline_confusion, summarize
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="stub", choices=["stub", "ollama", "openai", "anthropic"])
    parser.add_argument("--model", default=None, help="Model name, if the provider needs one")
    parser.add_argument("--n-users", type=int, default=12, help="Synthetic users to generate")
    parser.add_argument("--n-negatives", type=int, default=1, help="Negative pairs per positive")
    args = parser.parse_args()

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
