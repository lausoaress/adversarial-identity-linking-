"""Metrics for comparing the baseline against the adversarial pipeline.

Four numbers, matching the protocol in the TCC doc's "Metodologia e
Dados" section:

  1. false positive rate   — primary metric
  2. recall                — must not collapse while FP rate drops
  3. average contestation rounds — cost of the adversarial layer
     (currently always 1; becomes meaningful once multi-round is built)
  4. unclear-resolution rate — the TCC's actual headline comparison: of
     the pairs the baseline's own real-world experiment would leave
     ambiguous, how many does the adversarial layer resolve (accept or
     reject, with a cited justification) instead of leaving ambiguous?

Everything here takes plain lists of (prediction, ground_truth) rather
than PipelineResult objects directly, so it's easy to unit test and easy
to point at either the baseline or the adversarial pipeline's output.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.schemas import PipelineResult, Verdict


@dataclass
class ConfusionCounts:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    unclear: int = 0  # only meaningful for the adversarial pipeline

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
            + self.unclear
        )

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else float("nan")

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positive + self.true_negative
        return self.false_positive / denom if denom else float("nan")

    @property
    def unclear_rate(self) -> float:
        return self.unclear / self.total if self.total else float("nan")


def baseline_confusion(decisions: list[tuple[bool, bool]]) -> ConfusionCounts:
    """`decisions` is a list of (accepted, ground_truth_same_person)."""
    counts = ConfusionCounts()
    for accepted, same_person in decisions:
        if accepted and same_person:
            counts.true_positive += 1
        elif accepted and not same_person:
            counts.false_positive += 1
        elif not accepted and same_person:
            counts.false_negative += 1
        else:
            counts.true_negative += 1
    return counts


def adversarial_confusion(results: list[PipelineResult]) -> ConfusionCounts:
    counts = ConfusionCounts()
    for r in results:
        if r.ground_truth is None:
            continue
        if r.juiz.verdict == Verdict.ACCEPT and r.ground_truth:
            counts.true_positive += 1
        elif r.juiz.verdict == Verdict.ACCEPT and not r.ground_truth:
            counts.false_positive += 1
        elif r.juiz.verdict == Verdict.REJECT and r.ground_truth:
            counts.false_negative += 1
        elif r.juiz.verdict == Verdict.REJECT and not r.ground_truth:
            counts.true_negative += 1
        else:
            counts.unclear += 1
    return counts


def average_rounds(results: list[PipelineResult]) -> float:
    if not results:
        return float("nan")
    return sum(r.juiz.rounds_used for r in results) / len(results)


def unclear_resolution_rate(
    baseline_unclear_pairs: list,  # pairs the baseline left ambiguous (e.g. near-threshold)
    adversarial_results_for_same_pairs: list[PipelineResult],
) -> float:
    """Of the pairs the baseline could not confidently decide, what
    fraction does the adversarial pipeline resolve (ACCEPT or REJECT,
    not UNCLEAR)? This is the number that most directly answers the TCC's
    research question — see the doc's "Pergunta de pesquisa" section.
    """
    if not baseline_unclear_pairs:
        return float("nan")
    resolved = sum(
        1 for r in adversarial_results_for_same_pairs if r.juiz.verdict != Verdict.UNCLEAR
    )
    return resolved / len(baseline_unclear_pairs)


def summarize(label: str, counts: ConfusionCounts) -> str:
    return (
        f"{label}: precision={counts.precision:.3f} recall={counts.recall:.3f} "
        f"fp_rate={counts.false_positive_rate:.3f} "
        f"unclear_rate={counts.unclear_rate:.3f} (n={counts.total})"
    )
