"""The Contestador agent.

Takes the Proposer's hypothesis and actively tries to break it — "Opção A"
from the TCC article (Seção 3.2): an LLM instructed to search for
alternative explanations and contrary evidence, rather than a second
stylometric classifier.

The prompt is deliberately built around the specific failure mode
documented in the baseline paper itself (Arabnezhad et al., 2020, Section
VI): of the real-world candidate pairs they manually inspected, the false
positives were almost all caught by a FACTUAL contradiction the
stylometric model had no way to see — mismatched stated age, religion,
political stance, geography, or habits. That is exactly the kind of
evidence a pure similarity score is blind to and an LLM reading the text
can look for directly. So the Contestador's job is explicitly split in
two: (1) look for factual contradictions, and (2) only then fall back to
stylistic doubts. This also keeps the Contestador from being a mere echo
of the Proposer (the risk Nguyen et al., 2025 flag) — it is looking for a
structurally different kind of signal, not re-scoring the same one.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.llm_client import LLMClient, safe_json_parse
from src.schemas import CandidatePair, ContestadorOutput, Evidence, ProposerOutput

SYSTEM_PROMPT = """\
You are the Contestador (Challenger) agent in a forensic identity-linking \
pipeline. The Proposer agent has hypothesized that two aliases are the \
same person. Your job is to actively try to break that hypothesis — not \
to rubber-stamp it, and not to just restate the Proposer's own evidence \
in different words.

Search for evidence IN THIS PRIORITY ORDER:

1. FACTUAL CONTRADICTIONS — stated facts that cannot both be true of one \
person: age, location/country, religion, political stance, relationship \
status, drug use or other habits, profession, or any other concrete \
self-reported detail. This is the single most reliable signal for \
ruling out a false match, because a similarity score cannot see it at \
all — two people can "sound alike" stylistically while one explicitly \
says they are 20 and the other says 34.

2. ALTERNATIVE EXPLANATIONS — could the stylistic similarity come from \
something other than shared identity (same community jargon, a common \
template, translated/paraphrased text, coincidence given a short sample)?

3. STYLISTIC DOUBTS — only if you find nothing in (1) or (2): specific \
stylistic inconsistencies the Proposer's evidence overlooked.

If you genuinely find nothing to contest after checking all three, say so \
plainly — do not invent a weak objection just to have one.

Respond with a single JSON object:
{
  "stance": "contests" | "supports" | "inconclusive",
  "rebuttal": "<your reasoning, referencing the specific evidence you found or did not find>",
  "evidence": [
    {"kind": "factual_contradiction" | "alternative_explanation" | "stylistic_doubt" | "factual_support",
     "description": "<specific, quotable observation>", "weight": <float 0..1>}
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Proposer's hypothesis: {hypothesis}
Proposer's confidence: {confidence:.2f}
Proposer's evidence:
{evidence_list}

Dark web alias "{dark_id}" — sample of their messages:
---
{dark_sample}
---

Open web alias "{open_id}" — sample of their messages:
---
{open_sample}
---

Try to refute the hypothesis. Check for factual contradictions first.
"""


@dataclass
class ContestadorAgent:
    llm: LLMClient
    sample_chars: int = 2000

    def run(self, pair: CandidatePair, proposer_output: ProposerOutput) -> ContestadorOutput:
        evidence_list = "\n".join(
            f"- [{e.kind}] {e.description} (weight={e.weight})" for e in proposer_output.evidence
        )
        user_prompt = USER_PROMPT_TEMPLATE.format(
            hypothesis=proposer_output.hypothesis,
            confidence=proposer_output.confidence,
            evidence_list=evidence_list or "(none)",
            dark_id=pair.dark_alias.alias_id,
            dark_sample=pair.dark_alias.text[: self.sample_chars],
            open_id=pair.open_alias.alias_id,
            open_sample=pair.open_alias.text[: self.sample_chars],
        )
        raw = self.llm.complete(SYSTEM_PROMPT, user_prompt, json_mode=True)
        parsed = safe_json_parse(raw)

        evidence = [
            Evidence(
                kind=item.get("kind", "factual_contradiction"),
                description=item["description"],
                weight=float(item.get("weight", 0.5)),
                source_agent="contestador",
            )
            for item in parsed.get("evidence", [])
        ]

        stance = parsed.get("stance", "inconclusive")
        if stance not in {"contests", "supports", "inconclusive"}:
            stance = "inconclusive"

        return ContestadorOutput(
            pair=pair,
            rebuttal=parsed.get("rebuttal", ""),
            evidence=evidence,
            stance=stance,
        )
