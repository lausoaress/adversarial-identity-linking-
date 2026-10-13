"""Smoke tests for the three agents using StubLLMClient — no real model
needed. These check that each agent's prompt-building and JSON-parsing
code actually works end to end, not that the reasoning is good (that
needs a real model and real data — see src/evaluation/run_experiment.py).
"""
import json

from src.agents.contestador import ContestadorAgent
from src.agents.juiz import JuizAgent
from src.agents.llm_client import StubLLMClient
from src.agents.orchestrator import AdversarialPipeline
from src.agents.proposer import ProposerAgent
from src.schemas import Alias, CandidatePair, Platform, Verdict


def _pair() -> CandidatePair:
    dark = Alias(alias_id="dark1", platform=Platform.DARK_WEB, source="test", messages=["hello world"])
    open_ = Alias(alias_id="open1", platform=Platform.OPEN_WEB, source="test", messages=["hello world"])
    return CandidatePair(dark_alias=dark, open_alias=open_, similarity_score=0.55)


def test_proposer_parses_scripted_response_and_logs_baseline_score():
    llm = StubLLMClient(
        scripted_responses=[
            json.dumps(
                {
                    "hypothesis": "Same person.",
                    "confidence": 0.7,
                    "evidence": [{"description": "shared phrase", "weight": 0.5}],
                }
            )
        ]
    )
    agent = ProposerAgent(llm=llm)
    result = agent.run(_pair())

    assert result.hypothesis == "Same person."
    assert result.confidence == 0.7
    # the baseline score is always prepended as evidence[0]
    assert result.evidence[0].source_agent == "baseline"
    assert "0.5500" in result.evidence[0].description
    assert result.evidence[1].description == "shared phrase"


def test_contestador_defaults_bad_stance_to_inconclusive():
    llm = StubLLMClient(
        scripted_responses=[
            json.dumps({"stance": "not-a-real-stance", "rebuttal": "hmm", "evidence": []})
        ]
    )
    from src.schemas import ProposerOutput

    proposer_output = ProposerOutput(pair=_pair(), hypothesis="h", evidence=[], confidence=0.5)
    agent = ContestadorAgent(llm=llm)
    result = agent.run(_pair(), proposer_output)
    assert result.stance == "inconclusive"


def test_juiz_falls_back_to_unclear_on_bad_verdict_string():
    llm = StubLLMClient(
        scripted_responses=[
            json.dumps(
                {"verdict": "maybe??", "explanation": "unsure", "cited_evidence_descriptions": []}
            )
        ]
    )
    from src.schemas import ContestadorOutput, ProposerOutput

    proposer_output = ProposerOutput(pair=_pair(), hypothesis="h", evidence=[], confidence=0.5)
    contestador_output = ContestadorOutput(pair=_pair(), rebuttal="r", evidence=[], stance="inconclusive")
    agent = JuizAgent(llm=llm)
    result = agent.run(_pair(), proposer_output, contestador_output)
    assert result.verdict == Verdict.UNCLEAR


def test_full_pipeline_runs_end_to_end_with_stub_llm():
    responses = [
        json.dumps({"hypothesis": "h", "confidence": 0.6, "evidence": []}),
        json.dumps({"stance": "supports", "rebuttal": "no issues found", "evidence": []}),
        json.dumps({"verdict": "accept", "explanation": "e", "cited_evidence_descriptions": []}),
    ]
    llm = StubLLMClient(scripted_responses=responses)
    pipeline = AdversarialPipeline.from_single_client(llm)
    result = pipeline.run(_pair())

    assert result.juiz.verdict == Verdict.ACCEPT
    assert llm.call_count == 3  # exactly one call per agent, no retries hiding a bug
