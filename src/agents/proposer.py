"""The Proposer agent.

Takes a candidate pair (typically one that already survived the baseline's
k-attribution stage — see src/baseline/cosine_pipeline.py) and produces a
structured hypothesis: why these two aliases might be the same person,
with explicit supporting evidence, not just a bare similarity score.

This is "Opção B" from the TCC article (Seção 3.1): an LLM reasoning over
the evidence, rather than a second, separately-trained stylometric
classifier (Opção A) — chosen because it gives the Contestador something
concrete to contest, instead of a single opaque number.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.llm_client import LLMClient, safe_json_parse
from src.schemas import CandidatePair, Evidence, ProposerOutput

SYSTEM_PROMPT = """\
You are the Proposer agent in a forensic identity-linking pipeline. You \
are given two text samples from two online aliases — one from a dark web \
forum/market, one from the open web — plus a stylometric similarity score \
from a baseline classifier. Your job is to propose whether they could be \
the same person, and to say exactly WHY: cite specific phrases, topics, \
writing habits, or posting-time patterns that support the hypothesis.

Be concrete. "The writing style is similar" is not useful; "both aliases \
use the phrase 'tbh' at the start of a sentence and reference the same \
niche topic" is useful, because it gives the next agent (the Contestador) \
something specific to try to refute.

Respond with a single JSON object:
{
  "hypothesis": "<one or two sentences stating the proposed link>",
  "confidence": <float 0..1>,
  "evidence": [
    {"description": "<specific, quotable observation>", "weight": <float 0..1>}
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Baseline stylometric similarity score: {similarity_score:.4f} (threshold for \
the baseline's own accept decision: {threshold:.4f})

Dark web alias "{dark_id}" — sample of their messages:
---
{dark_sample}
---

Open web alias "{open_id}" — sample of their messages:
---
{open_sample}
---

Propose whether these are the same person.
"""


@dataclass
class ProposerAgent:
    llm: LLMClient
    sample_chars: int = 2000  # keep prompts bounded; raise once you're past smoke-testing

    def run(self, pair: CandidatePair, threshold: float = 0.4190) -> ProposerOutput:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            similarity_score=pair.similarity_score or 0.0,
            threshold=threshold,
            dark_id=pair.dark_alias.alias_id,
            dark_sample=pair.dark_alias.text[: self.sample_chars],
            open_id=pair.open_alias.alias_id,
            open_sample=pair.open_alias.text[: self.sample_chars],
        )
        raw = self.llm.complete(SYSTEM_PROMPT, user_prompt, json_mode=True)
        parsed = safe_json_parse(raw)

        evidence = [
            Evidence(
                kind="stylometric",
                description=item["description"],
                weight=float(item.get("weight", 0.5)),
                source_agent="proposer",
            )
            for item in parsed.get("evidence", [])
        ]
        # The baseline score itself is always logged as evidence too, so the
        # Juiz can see exactly what the single-agent baseline would have
        # decided, alongside what the Proposer adds on top of it.
        evidence.insert(
            0,
            Evidence(
                kind="stylometric",
                description=f"Baseline cosine similarity = {pair.similarity_score:.4f}",
                weight=pair.similarity_score,
                source_agent="baseline",
                raw={"similarity_score": pair.similarity_score},
            ),
        )

        return ProposerOutput(
            pair=pair,
            hypothesis=parsed.get("hypothesis", ""),
            evidence=evidence,
            confidence=float(parsed.get("confidence", 0.5)),
        )
