"""Tests for the Proposer's verdict rule, quote check, sampling and
handling of unusable replies. All use StubLLMClient — no model needed.
"""
import json

from src.agents.llm_client import StubLLMClient
from src.agents.proposer import (
    ProposerAgent,
    activity_summary,
    decide_verdict,
    is_grounded,
    sample_messages,
)
from src.schemas import Alias, CandidatePair, Platform, ProposerVerdict

DARK_TEXT = ["tbh the vendor shipped fast", "I live in Miami and love yoga", "second message here"]
OPEN_TEXT = ["tbh i got my order quick", "Living in Miami is great", "another open message"]


def _pair(score=0.55) -> CandidatePair:
    dark = Alias("dark1", Platform.DARK_WEB, "test", DARK_TEXT[:], [1, 2, 3])
    open_ = Alias("open1", Platform.OPEN_WEB, "test", OPEN_TEXT[:], [13, 14, 15])
    return CandidatePair(dark, open_, similarity_score=score)


def _reply(**overrides) -> str:
    body = {
        "verdict": "link",
        "confidence": 0.8,
        "hypothesis": "Same person.",
        "evidence": [
            {
                "kind": "factual_support",
                "description": "both say they live in Miami",
                "quote_dark": "I live in Miami",
                "quote_open": "Living in Miami",
                "weight": 0.9,
            }
        ],
    }
    body.update(overrides)
    return json.dumps(body)


def _run(*replies, pair=None, **agent_kwargs):
    llm = StubLLMClient(scripted_responses=list(replies))
    out = ProposerAgent(llm=llm, **agent_kwargs).run(pair or _pair())
    return out, llm


# --- verdict rule (DECISÃO 1) ------------------------------------------------

def test_link_with_verified_evidence_stays_link():
    out, _ = _run(_reply())
    assert out.verdict == ProposerVerdict.LINK
    assert out.evidence[0].source_agent == "baseline"
    assert out.evidence[1].raw["grounded"] is True


def test_link_without_verified_evidence_is_downgraded_to_unsure():
    reply = _reply(
        evidence=[{"description": "invented", "quote_dark": "never said this", "quote_open": "", "weight": 0.9}]
    )
    out, _ = _run(reply)
    assert out.verdict == ProposerVerdict.UNSURE


def test_verdict_is_derived_from_confidence_when_model_gives_none():
    assert decide_verdict(None, 0.9, n_grounded=1) == ProposerVerdict.LINK
    assert decide_verdict("maybe??", 0.1, n_grounded=0) == ProposerVerdict.NO_LINK
    assert decide_verdict(None, 0.5, n_grounded=1) == ProposerVerdict.UNSURE
    assert decide_verdict(None, 0.9, n_grounded=0) == ProposerVerdict.UNSURE


def test_no_link_and_unsure_pass_through():
    assert decide_verdict("no_link", 0.9, n_grounded=0) == ProposerVerdict.NO_LINK
    assert decide_verdict("UNSURE", 0.9, n_grounded=3) == ProposerVerdict.UNSURE


# --- quote check (DECISÃO 2) -------------------------------------------------

def test_quote_check_ignores_case_and_whitespace():
    assert is_grounded("I  LIVE in\nmiami", "", "so i live in Miami now", "")


def test_quote_check_needs_every_given_quote_to_match():
    assert not is_grounded("I live in Miami", "made up quote", "I live in Miami", "other text")


def test_quote_check_rejects_missing_or_tiny_quotes():
    assert not is_grounded("", "", "anything", "anything")
    assert not is_grounded("a", "", "a cat", "")


def test_unverified_evidence_is_dropped_but_recorded():
    reply = _reply(
        evidence=[
            {"description": "real", "quote_dark": "I live in Miami", "quote_open": "", "weight": 0.5},
            {"description": "invented", "quote_dark": "not in text", "quote_open": "", "weight": 0.5},
        ]
    )
    out, _ = _run(reply)
    descriptions = [e.description for e in out.evidence]
    assert "real" in descriptions and "invented" not in descriptions
    assert [d["description"] for d in out.dropped_evidence] == ["invented"]


def test_temporal_evidence_quoting_the_activity_line_is_kept():
    # the pair's dark alias posts at hours 1,2,3 -> "00-05h: 100%"
    reply = _reply(
        verdict="unsure",
        evidence=[
            {"kind": "temporal", "description": "dark alias posts at night",
             "quote_dark": "00-05h: 100%", "quote_open": "", "weight": 0.4}
        ],
    )
    out, _ = _run(reply)
    assert any(e.kind == "temporal" for e in out.evidence)
    assert out.dropped_evidence == []


def test_temporal_evidence_with_a_false_quote_is_dropped():
    reply = _reply(
        verdict="unsure",
        evidence=[
            {"kind": "temporal", "description": "both post in the evening",
             "quote_dark": "18-23h: 90%", "quote_open": "", "weight": 0.4}
        ],
    )
    out, _ = _run(reply)
    assert not any(e.kind == "temporal" for e in out.evidence)
    assert len(out.dropped_evidence) == 1


def test_temporal_evidence_without_any_quote_is_dropped():
    reply = _reply(
        verdict="unsure",
        evidence=[{"kind": "temporal", "description": "both post in the evening", "weight": 0.4}],
    )
    out, _ = _run(reply)
    assert len(out.dropped_evidence) == 1


def test_unknown_evidence_kind_falls_back_to_stylometric():
    reply = _reply(
        evidence=[
            {"kind": "vibes", "description": "x", "quote_dark": "I live in Miami", "quote_open": "", "weight": 0.5}
        ]
    )
    out, _ = _run(reply)
    assert out.evidence[1].kind == "stylometric"


# --- sampling (DECISÃO 3) ----------------------------------------------------

def _long_alias() -> Alias:
    msgs = [f"message number {i} " + "word " * 20 for i in range(40)]
    return Alias("long1", Platform.DARK_WEB, "test", msgs, [i % 24 for i in range(40)])


def test_sample_is_deterministic_and_within_budget():
    a = _long_alias()
    s1 = sample_messages(a, budget_chars=500, strategy="random", seed=1)
    s2 = sample_messages(a, budget_chars=500, strategy="random", seed=1)
    assert s1 == s2
    assert len(s1) <= 500


def test_sample_changes_with_seed_and_keeps_original_order():
    a = _long_alias()
    s1 = sample_messages(a, budget_chars=500, strategy="random", seed=1)
    s2 = sample_messages(a, budget_chars=500, strategy="random", seed=2)
    assert s1 != s2
    numbers = [int(line.split()[2]) for line in s1.splitlines() if line.startswith("message number")]
    assert numbers == sorted(numbers)


def test_first_strategy_takes_the_start():
    a = _long_alias()
    s = sample_messages(a, budget_chars=200, strategy="first")
    assert s.startswith("message number 0 ")


def test_unknown_strategy_raises():
    try:
        sample_messages(_long_alias(), strategy="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_activity_summary_reports_share_per_block():
    a = Alias("x", Platform.DARK_WEB, "t", ["m"] * 4, [0, 1, 13, 20])
    text = activity_summary(a)
    assert "00-05h: 50%" in text and "12-17h: 25%" in text and "18-23h: 25%" in text


def test_activity_summary_handles_no_timestamps():
    assert "no timestamps" in activity_summary(Alias("x", Platform.DARK_WEB, "t", ["m"]))


def test_prompt_contains_activity_and_sample():
    _, llm = _run(_reply())
    user_prompt = llm._calls[0]["user"]
    assert "Posting activity:" in user_prompt
    assert "tbh the vendor shipped fast" in user_prompt


def test_baseline_score_can_be_hidden_for_ablation():
    _, llm = _run(_reply(), show_baseline_score=False)
    assert "0.5500" not in llm._calls[0]["user"]


# --- unusable replies (DECISÃO 4) --------------------------------------------

def test_missing_baseline_score_does_not_crash():
    out, _ = _run(_reply(), pair=_pair(score=None))
    assert "n/a" in out.evidence[0].description


def test_evidence_item_without_description_is_skipped_with_warning():
    reply = _reply(evidence=[{"weight": 0.3}])
    out, _ = _run(reply)
    assert not out.parse_error
    assert any("without a description" in w for w in out.warnings)


def test_non_numeric_confidence_is_replaced_and_warned():
    out, _ = _run(_reply(confidence="high"))
    assert out.confidence == 0.5
    assert any("not a number" in w for w in out.warnings)


def test_confidence_is_clamped_to_unit_interval():
    out, _ = _run(_reply(confidence=7))
    assert out.confidence == 1.0


def test_prose_reply_is_retried_and_second_reply_is_used():
    out, llm = _run("Sure, they look similar.", _reply())
    assert llm.call_count == 2
    assert not out.parse_error
    assert out.verdict == ProposerVerdict.LINK
    assert "could not be used" in llm._calls[1]["user"]


def test_two_unusable_replies_give_a_marked_error_not_a_crash():
    out, llm = _run("no json here", "still no json")
    assert llm.call_count == 2
    assert out.parse_error is True
    assert out.verdict == ProposerVerdict.UNSURE
    assert out.evidence[0].source_agent == "baseline"
    assert out.raw_response == "still no json"
