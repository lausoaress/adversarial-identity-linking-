"""Tests for the pool-based evaluation protocol (src/evaluation/protocol.py).
No model needed: the hand-built cases use made-up scores, and the end-to-end
cases use the synthetic corpus and StubLLMClient.
"""
import json
import math

from src.agents.llm_client import StubLLMClient
from src.agents.orchestrator import AdversarialPipeline
from src.baseline.cosine_pipeline import BaselineConfig
from src.data.loaders import make_synthetic_corpus
from src.evaluation.protocol import (
    ACCEPT,
    REJECT,
    UNCLEAR,
    PoolCounts,
    Query,
    ScoredQuery,
    baseline_decisions,
    bootstrap_ci,
    build_pool_task,
    calibrate_threshold,
    classify,
    compare_arms,
    count_outcomes,
    low_threshold,
    paired_sign_test,
    resolution_rate,
    run_pool_protocol,
    zone,
)
from src.schemas import Alias, CandidatePair, Platform


def _alias(i: str) -> Alias:
    return Alias(i, Platform.OPEN_WEB, "t", ["x"], [1])


def _sq(score, truth="a", best_id="a", has_best=True) -> ScoredQuery:
    """A scored query with a made-up best candidate. truth=None -> no-match query."""
    q = Query(_alias("q"), truth)
    best = CandidatePair(_alias("q"), _alias(best_id), similarity_score=score) if has_best else None
    return ScoredQuery(q, best, in_top_k=truth is not None and best_id == truth)


# --- outcome classification -----------------------------------------------------

def test_classify_all_cases():
    right = _sq(0.6)                    # matchable, best is the true match
    wrong = _sq(0.6, best_id="other")   # matchable, best is someone else
    nomatch = _sq(0.6, truth=None)      # no true match exists
    assert classify(right, ACCEPT) == "TP"
    assert classify(wrong, ACCEPT) == "FP"      # accepted the wrong person
    assert classify(nomatch, ACCEPT) == "FP"
    assert classify(right, REJECT) == "FN"
    assert classify(wrong, REJECT) == "FN"      # a true match existed and was not recovered
    assert classify(nomatch, REJECT) == "TN"
    assert classify(right, UNCLEAR) == "UNCLEAR"


def test_counts_and_rates():
    scored = [_sq(0.6), _sq(0.6, best_id="o"), _sq(0.6, truth=None), _sq(0.2, truth=None)]
    c = count_outcomes(scored, [ACCEPT, ACCEPT, ACCEPT, REJECT])
    assert (c.tp, c.fp, c.fn, c.tn, c.unclear) == (1, 2, 0, 1, 0)
    assert c.n_matchable == 2 and c.n_queries == 4
    assert math.isclose(c.precision, 1 / 3)
    assert math.isclose(c.recall, 1 / 2)        # over ALL matchable queries
    assert math.isclose(c.fp_per_query, 2 / 4)


def test_empty_counts_give_nan_not_a_crash():
    c = PoolCounts()
    assert math.isnan(c.precision) and math.isnan(c.recall) and math.isnan(c.fn_rate)


def test_query_the_baseline_could_not_score_is_rejected_and_counts_as_fn():
    sq = _sq(0.0, has_best=False)
    assert baseline_decisions([sq], 0.419) == [REJECT]
    assert classify(sq, REJECT) == "FN"


# --- thresholds -----------------------------------------------------------------

def test_calibrated_threshold_reaches_target_recall_on_w1():
    # 10 matchable queries, all correct, scores 0.1 ... 1.0
    w1 = [_sq(s / 10) for s in range(1, 11)]
    t = calibrate_threshold(w1, target_recall=0.80)
    decisions = baseline_decisions(w1, t)
    assert count_outcomes(w1, decisions).recall >= 0.80
    assert math.isclose(t, 0.3)                 # the 8th best score


def test_calibration_falls_back_when_w1_has_no_correct_candidate():
    w1 = [_sq(0.5, best_id="other"), _sq(0.5, truth=None)]
    assert calibrate_threshold(w1, fallback=0.4190) == 0.4190


def test_low_threshold_covers_the_requested_share_of_correct_candidates():
    w1 = [_sq(s / 100) for s in range(1, 101)]  # 100 correct, 0.01 ... 1.00
    t = low_threshold(w1, coverage=0.95)
    covered = sum(sq.score >= t for sq in w1) / len(w1)
    assert covered >= 0.95
    assert t <= 0.06


def test_zones():
    assert zone(_sq(0.10), 0.2, 0.4) == "below"
    assert zone(_sq(0.30), 0.2, 0.4) == "rescue_zone"
    assert zone(_sq(0.40), 0.2, 0.4) == "accept_zone"
    assert zone(_sq(0.0, has_best=False), 0.2, 0.4) == "below"


# --- uncertainty ----------------------------------------------------------------

def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    scored = [_sq(0.6) for _ in range(8)] + [_sq(0.6, best_id="o") for _ in range(2)]
    decisions = [ACCEPT] * 10
    lo1, hi1 = bootstrap_ci(scored, decisions, "precision", n_boot=200)
    lo2, hi2 = bootstrap_ci(scored, decisions, "precision", n_boot=200)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= 0.8 <= hi1


def test_sign_test_known_values():
    # B right where A is wrong on 8 queries, the opposite on 0 -> p = 2 * 0.5^8
    b_only, a_only, p = paired_sign_test([False] * 8, [True] * 8)
    assert (b_only, a_only) == (8, 0)
    assert math.isclose(p, 2 * 0.5 ** 8)
    # a tie of differences gives p = 1
    assert paired_sign_test([True, False], [False, True])[2] == 1.0
    # no differing queries at all
    assert paired_sign_test([True, True], [True, True]) == (0, 0, 1.0)


# --- task construction ----------------------------------------------------------

def _corpus(n=20):
    return make_synthetic_corpus(n_users=n, platform=Platform.OPEN_WEB)


def test_pool_task_is_deterministic():
    t1 = build_pool_task(_corpus(), seed=3)
    t2 = build_pool_task(_corpus(), seed=3)
    assert [a.alias_id for a in t1.pool] == [a.alias_id for a in t2.pool]
    assert [q.alias.alias_id for q in t1.w2] == [q.alias.alias_id for q in t2.w2]


def test_no_match_queries_have_no_alias_in_the_pool():
    task = build_pool_task(_corpus(), no_match_frac=0.25)
    pool_ids = {a.alias_id for a in task.pool}
    queries = task.w1 + task.w2
    no_match = [q for q in queries if q.truth_id is None]
    matchable = [q for q in queries if q.truth_id is not None]
    assert no_match and matchable
    for q in matchable:
        assert q.truth_id in pool_ids
    for q in no_match:
        # the query's own source user must be absent from the pool
        source = q.alias.alias_id.replace("__alter", "__orig")
        assert source not in pool_ids
    assert len(task.pool) == len(matchable)


def test_w1_and_w2_do_not_share_queries_and_both_have_both_kinds():
    task = build_pool_task(_corpus(24), no_match_frac=0.25, calib_frac=0.5)
    ids1 = {q.alias.alias_id for q in task.w1}
    ids2 = {q.alias.alias_id for q in task.w2}
    assert not ids1 & ids2
    for part in (task.w1, task.w2):
        assert any(q.truth_id is None for q in part)
        assert any(q.truth_id is not None for q in part)


def test_zero_no_match_fraction_gives_only_matchable_queries():
    task = build_pool_task(_corpus(), no_match_frac=0.0)
    assert all(q.truth_id is not None for q in task.w1 + task.w2)


# --- end to end ------------------------------------------------------------------

def test_baseline_only_run_has_two_arms_and_consistent_sizes():
    task = build_pool_task(_corpus(20))
    res = run_pool_protocol(task, BaselineConfig(), pipeline=None)
    assert set(res.arms) == {"baseline (paper threshold)", "baseline (calibrated on W1)"}
    for decisions in res.arms.values():
        assert len(decisions) == len(task.w2) == len(res.w2) == len(res.zones)
    assert res.t_low <= res.t_paper


def _stub_pipeline(n_calls: int) -> AdversarialPipeline:
    one = [
        json.dumps({"verdict": "no_link", "confidence": 0.3, "hypothesis": "h", "evidence": []}),
        json.dumps({"stance": "inconclusive", "rebuttal": "r", "evidence": []}),
        json.dumps({"verdict": "reject", "explanation": "e", "cited_evidence_descriptions": []}),
    ]
    return AdversarialPipeline.from_single_client(StubLLMClient(scripted_responses=one * n_calls))


def test_llm_runs_only_outside_the_below_zone_and_arms_line_up():
    task = build_pool_task(_corpus(20))
    pipe = _stub_pipeline(len(task.w2) + 2)
    res = run_pool_protocol(task, BaselineConfig(), pipeline=pipe)
    in_band = [i for i, z in enumerate(res.zones) if z != "below"]
    assert sorted(res.llm_results) == in_band
    assert set(res.arms) >= {"Proposer alone", "full pipeline"}
    for i, z in enumerate(res.zones):
        if z == "below":
            assert res.arms["full pipeline"][i] == REJECT
            assert res.arms["Proposer alone"][i] == REJECT
    # the stub always says no_link / reject
    for i in in_band:
        assert res.arms["full pipeline"][i] == REJECT
        assert res.arms["Proposer alone"][i] == REJECT


# --- comparison with the baseline ---------------------------------------------------

def test_compare_arms_counts_each_kind_of_change():
    scored = [
        _sq(0.6, best_id="o"),        # 0: baseline FP  -> arm rejects  => fp_removed
        _sq(0.3),                     # 1: baseline FN  -> arm accepts  => fn_rescued
        _sq(0.6),                     # 2: baseline TP  -> arm rejects  => tp_lost
        _sq(0.3, truth=None),         # 3: baseline TN  -> arm accepts  => fp_created
        _sq(0.6, best_id="o"),        # 4: baseline FP  -> arm unclear  => fp_removed
    ]
    ref = [ACCEPT, REJECT, ACCEPT, REJECT, ACCEPT]
    arm = [REJECT, ACCEPT, REJECT, ACCEPT, UNCLEAR]
    assert compare_arms(scored, ref, arm) == {
        "fp_removed": 2, "fn_rescued": 1, "tp_lost": 1, "fp_created": 1,
    }


def test_resolution_rate_only_looks_at_queries_that_reached_the_llm():
    decisions = [ACCEPT, UNCLEAR, REJECT, UNCLEAR]
    zones = ["accept_zone", "rescue_zone", "below", "accept_zone"]
    assert math.isclose(resolution_rate(decisions, zones), 1 / 3)
    assert math.isnan(resolution_rate([REJECT], ["below"]))
