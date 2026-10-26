"""Pool-based evaluation protocol, modelled on Arabnezhad et al. (2020).

What we keep from the reference paper:

  - Alter-ego ground truth: each user's messages are split in two halves.
    One half goes into the pool of KNOWN aliases, the other half is the
    UNKNOWN query. The true match is known by construction.
  - Retrieval, not 1-vs-1 pairs: each query is searched against the whole
    pool, stage 1 keeps the top k (k=10), stage 2 scores those k with
    cosine similarity.
  - Two query sets. The decision thresholds are chosen on W1 (calibration)
    and everything is reported on W2 (test), as the paper did when it
    picked 0.4190 on one set and checked it on another.
  - Precision and recall as the headline numbers. Recall is counted over
    ALL matchable queries, so a true match lost in stage 1 is a false
    negative too.

What we add, because the research question asks about both error types:

  - "No-match" queries: some queries have their true alias REMOVED from
    the pool. Any accepted candidate for them is a false positive, which
    is the realistic case (most dark web aliases have no open-web twin).
  - Two bands around the baseline threshold, calibrated on W1:
        score >= T_HIGH            baseline would accept  -> LLM pipeline may overturn it
        T_LOW <= score < T_HIGH    baseline would reject  -> LLM pipeline may rescue it
        score < T_LOW              rejected without calling the LLM
    So the pipeline can fix false positives AND false negatives, and the
    LLM only runs on a fraction of the queries.
  - Arms compared on the SAME W2 queries: baseline at the paper's
    threshold, baseline recalibrated on W1, the Proposer alone (the
    single-LLM-agent control) and the full Proposer/Contestador/Juiz.
  - Uncertainty: bootstrap confidence intervals and an exact paired sign
    test, since W2 is small.

Decisions taken here (change them in one place if you disagree):

  D1  T_HIGH is the paper's 0.4190; a W1-recalibrated baseline is reported
      next to it. T_LOW is the lowest score that still covers 95% of the
      correct best candidates in W1.
  D2  Only the best (top-1) candidate of each query goes to the LLM.
  D3  Arms are compared as single operating points against the baseline's
      precision-recall sweep, because the LLM pipeline gives decisions,
      not a calibrated continuous score.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from src.baseline.cosine_pipeline import BaselineConfig, run_baseline
from src.data.alter_ego import build_alter_ego_pairs
from src.schemas import Alias, CandidatePair

ACCEPT, REJECT, UNCLEAR = "accept", "reject", "unclear"


# --- task construction ---------------------------------------------------------

@dataclass
class Query:
    alias: Alias
    truth_id: Optional[str]  # alias_id of its true match in the pool; None = no match exists


@dataclass
class PoolTask:
    pool: list[Alias]
    w1: list[Query]  # calibration queries
    w2: list[Query]  # test queries


def build_pool_task(
    aliases: list[Alias],
    seed: int = 13,
    no_match_frac: float = 0.25,
    calib_frac: float = 0.5,
) -> PoolTask:
    """Alter-ego split of every eligible user, then: a fraction of users
    lose their pool alias (their query becomes a no-match query), and the
    queries are split into W1 / W2 keeping the match / no-match proportion.
    """
    pairs = build_alter_ego_pairs(aliases, seed=seed)
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    n_no_match = int(round(no_match_frac * len(pairs)))
    no_match = set(order[:n_no_match])

    pool = [p.original for i, p in enumerate(pairs) if i not in no_match]
    matchable = [Query(p.alter_ego, p.original.alias_id) for i, p in enumerate(pairs) if i not in no_match]
    unmatched = [Query(p.alter_ego, None) for i, p in enumerate(pairs) if i in no_match]

    w1: list[Query] = []
    w2: list[Query] = []
    for group in (matchable, unmatched):
        group = group[:]
        rng.shuffle(group)
        cut = int(round(calib_frac * len(group)))
        w1.extend(group[:cut])
        w2.extend(group[cut:])
    return PoolTask(pool=pool, w1=w1, w2=w2)


# --- baseline scoring ----------------------------------------------------------

@dataclass
class ScoredQuery:
    query: Query
    best: Optional[CandidatePair]  # top-1 after stage 2; None if the baseline could not score the query
    in_top_k: bool  # the true match survived stage 1 (always False for no-match queries)

    @property
    def score(self) -> float:
        return (self.best.similarity_score or 0.0) if self.best else 0.0

    @property
    def matchable(self) -> bool:
        return self.query.truth_id is not None

    @property
    def correct(self) -> bool:
        """The best candidate IS the true match."""
        return bool(self.best and self.matchable and self.best.open_alias.alias_id == self.query.truth_id)


def score_queries(queries: list[Query], pool: list[Alias], config: BaselineConfig) -> list[ScoredQuery]:
    scored = []
    for q in queries:
        pairs = run_baseline(q.alias, pool, config)  # sorted, best first
        in_top_k = q.truth_id is not None and any(p.open_alias.alias_id == q.truth_id for p in pairs)
        scored.append(ScoredQuery(q, pairs[0] if pairs else None, in_top_k))
    return scored


# --- thresholds (calibrated on W1) ---------------------------------------------

def calibrate_threshold(w1: list[ScoredQuery], target_recall: float = 0.80, fallback: float = 0.4190) -> float:
    """Largest threshold whose recall on W1 reaches `target_recall`
    (the paper anchors its threshold at ~80% recall).
    """
    n_matchable = sum(sq.matchable for sq in w1)
    correct = sorted((sq.score for sq in w1 if sq.correct), reverse=True)
    if n_matchable == 0 or not correct:
        return fallback
    need = max(1, math.ceil(target_recall * n_matchable))
    return correct[need - 1] if len(correct) >= need else correct[-1]


def low_threshold(w1: list[ScoredQuery], coverage: float = 0.95) -> float:
    """Lowest score that still covers `coverage` of the correct best
    candidates in W1. Below it, a true match is rare enough to skip the LLM.
    """
    correct = sorted(sq.score for sq in w1 if sq.correct)
    if not correct:
        return 0.0
    return correct[int(math.floor((1.0 - coverage) * len(correct)))]


# --- outcomes and metrics ------------------------------------------------------

def classify(sq: ScoredQuery, decision: str) -> str:
    """TP / FP / FN / TN / UNCLEAR for one query and one arm's decision."""
    if decision == UNCLEAR:
        return "UNCLEAR"
    if decision == ACCEPT:
        return "TP" if sq.correct else "FP"
    return "FN" if sq.matchable else "TN"


@dataclass
class PoolCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    unclear: int = 0
    n_matchable: int = 0
    n_queries: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self) -> float:
        # over ALL matchable queries: wrong accepts, rejects and unclear all count as not recovered
        return self.tp / self.n_matchable if self.n_matchable else float("nan")

    @property
    def fp_per_query(self) -> float:
        return self.fp / self.n_queries if self.n_queries else float("nan")

    @property
    def fn_rate(self) -> float:
        return self.fn / self.n_matchable if self.n_matchable else float("nan")

    @property
    def unclear_rate(self) -> float:
        return self.unclear / self.n_queries if self.n_queries else float("nan")


def count_outcomes(scored: list[ScoredQuery], decisions: list[str]) -> PoolCounts:
    c = PoolCounts(n_queries=len(scored), n_matchable=sum(sq.matchable for sq in scored))
    for sq, d in zip(scored, decisions):
        outcome = classify(sq, d).lower()  # 'tp' | 'fp' | 'fn' | 'tn' | 'unclear'
        setattr(c, outcome, getattr(c, outcome) + 1)
    return c


def baseline_decisions(scored: list[ScoredQuery], threshold: float) -> list[str]:
    return [ACCEPT if (sq.best is not None and sq.score >= threshold) else REJECT for sq in scored]


def zone(sq: ScoredQuery, t_low: float, t_high: float) -> str:
    if sq.best is None or sq.score < t_low:
        return "below"
    return "accept_zone" if sq.score >= t_high else "rescue_zone"


# --- uncertainty ---------------------------------------------------------------

def bootstrap_ci(
    scored: list[ScoredQuery],
    decisions: list[str],
    metric: str,
    n_boot: int = 1000,
    seed: int = 13,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap over queries for `metric` in {'precision', 'recall'}."""
    rng = random.Random(seed)
    n = len(scored)
    values = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = getattr(count_outcomes([scored[i] for i in idx], [decisions[i] for i in idx]), metric)
        if not math.isnan(v):
            values.append(v)
    if not values:
        return float("nan"), float("nan")
    values.sort()
    lo = values[int(((1 - level) / 2) * (len(values) - 1))]
    hi = values[int((1 - (1 - level) / 2) * (len(values) - 1))]
    return lo, hi


def query_correct(sq: ScoredQuery, decision: str) -> bool:
    """A right call: TP or TN. UNCLEAR counts as not right (conservative)."""
    return classify(sq, decision) in ("TP", "TN")


def paired_sign_test(a_right: list[bool], b_right: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided sign test on the queries where arms A and B differ.
    Returns (b_only_right, a_only_right, p_value).
    """
    b_only = sum(1 for a, b in zip(a_right, b_right) if b and not a)
    a_only = sum(1 for a, b in zip(a_right, b_right) if a and not b)
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b_only, a_only, min(1.0, 2 * p)


# --- the whole protocol --------------------------------------------------------

@dataclass
class PoolResult:
    t_paper: float
    t_calibrated: float
    t_low: float
    w2: list[ScoredQuery]
    arms: dict[str, list[str]] = field(default_factory=dict)  # arm name -> decision per W2 query
    zones: list[str] = field(default_factory=list)
    llm_results: dict[int, object] = field(default_factory=dict)  # W2 index -> PipelineResult


def run_pool_protocol(
    task: PoolTask,
    config: BaselineConfig,
    pipeline=None,  # an AdversarialPipeline, or None to run the baseline arms only
    target_recall: float = 0.80,
    low_coverage: float = 0.95,
) -> PoolResult:
    from src.schemas import ProposerVerdict, Verdict

    w1 = score_queries(task.w1, task.pool, config)
    w2 = score_queries(task.w2, task.pool, config)

    t_paper = config.threshold
    t_cal = calibrate_threshold(w1, target_recall, fallback=t_paper)
    t_low = min(low_threshold(w1, low_coverage), t_paper)

    result = PoolResult(t_paper=t_paper, t_calibrated=t_cal, t_low=t_low, w2=w2)
    result.zones = [zone(sq, t_low, t_paper) for sq in w2]
    result.arms["baseline (paper threshold)"] = baseline_decisions(w2, t_paper)
    result.arms["baseline (calibrated on W1)"] = baseline_decisions(w2, t_cal)

    if pipeline is not None:
        proposer_arm, full_arm = [], []
        for i, (sq, z) in enumerate(zip(w2, result.zones)):
            if z == "below":
                proposer_arm.append(REJECT)
                full_arm.append(REJECT)
                continue
            r = pipeline.run(sq.best, baseline_threshold=t_paper)
            r.ground_truth = sq.correct
            result.llm_results[i] = r
            pv = r.proposer.verdict
            proposer_arm.append(
                ACCEPT if pv == ProposerVerdict.LINK else REJECT if pv == ProposerVerdict.NO_LINK else UNCLEAR
            )
            jv = r.juiz.verdict
            full_arm.append(ACCEPT if jv == Verdict.ACCEPT else REJECT if jv == Verdict.REJECT else UNCLEAR)
        result.arms["Proposer alone"] = proposer_arm
        result.arms["full pipeline"] = full_arm
    return result


# --- what the pipeline changed relative to the baseline ---------------------------

def compare_arms(scored: list[ScoredQuery], ref: list[str], arm: list[str]) -> dict[str, int]:
    """Query-by-query changes of `arm` relative to `ref` (normally the
    baseline at the paper's threshold):

      fp_removed   ref accepted a wrong candidate, arm no longer accepts it
      fn_rescued   ref rejected a true match, arm accepts it (TP)
      tp_lost      ref had a TP, arm does not
      fp_created   ref did not accept, arm accepts a wrong candidate
    """
    out = {"fp_removed": 0, "fn_rescued": 0, "tp_lost": 0, "fp_created": 0}
    for sq, r, a in zip(scored, ref, arm):
        ro, ao = classify(sq, r), classify(sq, a)
        if ro == "FP" and a != ACCEPT:
            out["fp_removed"] += 1
        if ro == "FN" and ao == "TP":
            out["fn_rescued"] += 1
        if ro == "TP" and ao != "TP":
            out["tp_lost"] += 1
        if ro != "FP" and ao == "FP":
            out["fp_created"] += 1
    return out


def resolution_rate(decisions: list[str], zones: list[str]) -> float:
    """Of the queries that reached the LLM (zone != 'below'), the share that
    got a decisive answer (ACCEPT or REJECT, not UNCLEAR). This is the
    "unclear resolution" the proposal talks about, measured on the band.
    """
    in_band = [d for d, z in zip(decisions, zones) if z != "below"]
    if not in_band:
        return float("nan")
    return sum(d != UNCLEAR for d in in_band) / len(in_band)
