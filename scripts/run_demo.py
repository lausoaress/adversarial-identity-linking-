"""Smallest possible end-to-end run: one candidate pair, through the
baseline AND the adversarial pipeline, printed in full.

No model required — uses StubLLMClient with scripted responses, so this
is the first thing to run after cloning: if this works, the schemas, the
baseline, and the three agents' prompt/parse plumbing all actually wire
together. Swap StubLLMClient for build_client("ollama") once you've
pulled a local model (scripts/setup_ollama.sh) to see real agent output
instead of the canned responses below.
"""
from __future__ import annotations

import json
import os
import sys

# Allow `python scripts/run_demo.py` to work from the repo root without
# having to `export PYTHONPATH=.` first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.llm_client import StubLLMClient
from src.agents.orchestrator import AdversarialPipeline
from src.baseline.cosine_pipeline import BaselineConfig, accepts, run_baseline
from src.data.alter_ego import build_alter_ego_pairs
from src.data.loaders import make_synthetic_corpus
from src.schemas import Platform

SCRIPTED_RESPONSES = [
    # Proposer
    json.dumps(
        {
            "hypothesis": "Both aliases discuss the same niche topics and share "
            "distinctive vocabulary, suggesting the same author.",
            "confidence": 0.62,
            "evidence": [
                {"description": "Both use the term 'term7' repeatedly, an uncommon word", "weight": 0.6},
                {"description": "Message length and punctuation habits are similar", "weight": 0.4},
            ],
        }
    ),
    # Contestador
    json.dumps(
        {
            "stance": "inconclusive",
            "rebuttal": "No factual contradiction found in either sample. The shared "
            "vocabulary could reflect a shared community rather than one author, "
            "but there is no evidence actively against the hypothesis either.",
            "evidence": [
                {
                    "kind": "alternative_explanation",
                    "description": "Shared vocabulary may reflect community jargon rather than authorship",
                    "weight": 0.3,
                }
            ],
        }
    ),
    # Juiz
    json.dumps(
        {
            "verdict": "accept",
            "explanation": "The Proposer's vocabulary-overlap evidence is specific and "
            "the Contestador found no factual contradiction, only a weaker alternative "
            "explanation. Weighing both, the link is accepted.",
            "cited_evidence_descriptions": [
                "Both use the term 'term7' repeatedly, an uncommon word",
                "Shared vocabulary may reflect community jargon rather than authorship",
            ],
        }
    ),
]


def main() -> None:
    print("Generating a tiny synthetic corpus and one alter-ego pair...")
    aliases = make_synthetic_corpus(n_users=3, platform=Platform.OPEN_WEB)
    ae_pairs = build_alter_ego_pairs(aliases)
    if not ae_pairs:
        print("No pair survived the minimum thresholds — try again, synthetic data is random.")
        return
    pair_data = ae_pairs[0]

    print(f"\nPair: {pair_data.original.alias_id}  <->  {pair_data.alter_ego.alias_id}")

    print("\n--- Baseline (single classifier) ---")
    config = BaselineConfig()
    scored = run_baseline(pair_data.original, [pair_data.alter_ego], config)
    if not scored:
        print("Baseline found no candidates above the minimum activity threshold.")
        return
    pair = scored[0]
    print(f"Cosine similarity: {pair.similarity_score:.4f} (threshold {config.threshold})")
    print(f"Baseline decision: {'ACCEPT' if accepts(pair, config) else 'REJECT'}")

    print("\n--- Adversarial pipeline (Proposer -> Contestador -> Juiz) ---")
    llm = StubLLMClient(scripted_responses=list(SCRIPTED_RESPONSES))
    pipeline = AdversarialPipeline.from_single_client(llm)
    result = pipeline.run(pair, baseline_threshold=config.threshold)

    print(f"Proposer hypothesis : {result.proposer.hypothesis}")
    print(f"Proposer confidence : {result.proposer.confidence:.2f}")
    print(f"Contestador stance  : {result.contestador.stance}")
    print(f"Contestador rebuttal: {result.contestador.rebuttal}")
    print(f"Juiz verdict        : {result.juiz.verdict.value.upper()}")
    print(f"Juiz explanation    : {result.juiz.explanation}")
    print(f"Evidence cited      : {[e.description for e in result.juiz.cited_evidence]}")

    print(
        "\n(This used scripted/stub LLM responses, not a real model — "
        "run scripts/setup_ollama.sh, then re-run with --provider ollama "
        "via src/evaluation/run_experiment.py for real agent reasoning.)"
    )


if __name__ == "__main__":
    main()
