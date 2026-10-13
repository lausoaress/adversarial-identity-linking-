"""The Juiz (Judge) agent.

Takes the Proposer's hypothesis and the Contestador's rebuttal and decides:
accept, reject, or unclear. "Opção B" from the TCC article (Seção 3.3): a
single weighing round rather than multi-round debate (Opção A, Han et al.
style) — simpler to implement and audit first; multi-round is a natural
next iteration (see src/agents/orchestrator.py for where that would plug
in via `max_rounds`).

The explicit goal this agent is instructed toward is the TCC's actual
contribution target: the baseline paper's own real-world experiment left
43% of candidate pairs "Unclear" (see the project's doc, "Trabalhos
Relacionados"). This prompt does not forbid "unclear" — an honest unclear
beats a confident wrong answer — but it does require the Juiz to try to
resolve it first, specifically by checking whether the Contestador's
factual-contradiction evidence (the kind the baseline literally cannot
see) is strong enough to break the tie one way or the other. Measuring
how often that works is the headline experiment (src/evaluation/metrics.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.llm_client import LLMClient, safe_json_parse
from src.schemas import (
    CandidatePair,
    ContestadorOutput,
    Evidence,
    JuizOutput,
    ProposerOutput,
    Verdict,
)

SYSTEM_PROMPT = """\
You are the Juiz (Judge) agent in a forensic identity-linking pipeline. \
You receive a hypothesis from the Proposer agent and a rebuttal from the \
Contestador agent, and must decide: ACCEPT the link, REJECT it, or mark \
it UNCLEAR.

Weighing rules, in order of priority:

1. A factual contradiction from the Contestador (mismatched age, \
location, religion, politics, habits, or any other concrete \
self-reported detail) is STRONG evidence against the link — it should \
usually be decisive toward REJECT even if the stylometric similarity is \
high, because stylistic similarity cannot explain away a stated fact.

2. Strong, specific factual or stylistic support from the Proposer, with \
no credible contradiction found, supports ACCEPT.

3. Mark UNCLEAR only when, after weighing both sides, there genuinely \
isn't enough to decide either way — not as a default when the decision \
is merely effortful. If you mark UNCLEAR, say specifically what \
additional evidence would resolve it.

Your explanation MUST cite the specific evidence items (by their \
description) that drove the decision — a bare verdict with no citation \
is not acceptable; this decision needs to be auditable by a human \
reviewer afterward.

Respond with a single JSON object:
{
  "verdict": "accept" | "reject" | "unclear",
  "explanation": "<reasoning that explicitly cites evidence items>",
  "cited_evidence_descriptions": ["<description string, copied exactly from the evidence you were given>"]
}
"""

USER_PROMPT_TEMPLATE = """\
Proposer's hypothesis: {hypothesis}
Proposer's confidence: {confidence:.2f}
Proposer's evidence:
{proposer_evidence}

Contestador's stance: {stance}
Contestador's rebuttal: {rebuttal}
Contestador's evidence:
{contestador_evidence}

Decide: accept, reject, or unclear.
"""


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(none)"
    return "\n".join(f"- [{e.kind}] {e.description} (weight={e.weight})" for e in evidence)


@dataclass
class JuizAgent:
    llm: LLMClient

    def run(
        self,
        pair: CandidatePair,
        proposer_output: ProposerOutput,
        contestador_output: ContestadorOutput,
        rounds_used: int = 1,
    ) -> JuizOutput:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            hypothesis=proposer_output.hypothesis,
            confidence=proposer_output.confidence,
            proposer_evidence=_format_evidence(proposer_output.evidence),
            stance=contestador_output.stance,
            rebuttal=contestador_output.rebuttal,
            contestador_evidence=_format_evidence(contestador_output.evidence),
        )
        raw = self.llm.complete(SYSTEM_PROMPT, user_prompt, json_mode=True)
        parsed = safe_json_parse(raw)

        verdict_str = parsed.get("verdict", "unclear")
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.UNCLEAR

        all_evidence = {e.description: e for e in proposer_output.evidence + contestador_output.evidence}
        cited = [
            all_evidence[desc]
            for desc in parsed.get("cited_evidence_descriptions", [])
            if desc in all_evidence
        ]

        return JuizOutput(
            pair=pair,
            verdict=verdict,
            explanation=parsed.get("explanation", ""),
            cited_evidence=cited,
            rounds_used=rounds_used,
        )
