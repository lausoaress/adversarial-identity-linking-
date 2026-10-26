"""The Proposer agent.

Takes a candidate pair (typically one that already survived the baseline's
k-attribution stage — see src/baseline/cosine_pipeline.py) and produces a
structured call on it: a verdict (link / no_link / unsure), a confidence,
and explicit supporting evidence quoted from the two aliases' own text,
not just a bare similarity score.

This is "Opção B" from the TCC article (Seção 3.1): an LLM reasoning over
the evidence, rather than a second, separately-trained stylometric
classifier (Opção A) — chosen because it gives the Contestador something
concrete to contest, instead of a single opaque number.

The Proposer also works on its own, as the "single LLM agent" control arm
of the experiment (baseline vs. Proposer alone vs. full adversarial
pipeline). That is why it now emits a verdict of its own.

Every project-specific RULE in this file is isolated in a block marked
"DECISÃO n" so it can be reviewed and changed in one place:

  DECISÃO 1  how the final verdict is computed        -> decide_verdict()
  DECISÃO 2  what happens to evidence that fails the quote check
                                                      -> GROUNDING_POLICY, is_grounded()
  DECISÃO 3  which messages the model sees            -> sample_messages(), activity_summary()
  DECISÃO 4  what happens when the reply is unusable  -> MAX_ATTEMPTS, _failed_output()
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.agents.llm_client import LLMClient, safe_json_parse
from src.schemas import Alias, CandidatePair, Evidence, ProposerOutput, ProposerVerdict

# ---------------------------------------------------------------------------
# DECISÃO 1 — verdict rule
# ---------------------------------------------------------------------------
LINK_MIN_CONFIDENCE = 0.70      # confidence >= this  -> LINK   (only when the model gave no valid verdict)
NO_LINK_MAX_CONFIDENCE = 0.30   # confidence <= this  -> NO_LINK
MIN_GROUNDED_EVIDENCE_FOR_LINK = 1  # a LINK needs at least this many quote-verified evidence items


def decide_verdict(
    model_verdict: Optional[str], confidence: float, n_grounded: int
) -> ProposerVerdict:
    """Combine what the model said with what we could verify.

    1. Use the model's own verdict if it is one of link / no_link / unsure.
       Otherwise derive one from its confidence.
    2. A LINK with too little verified evidence is downgraded to UNSURE:
       the model cannot link two aliases on claims we could not find in
       the text.
    """
    try:
        verdict = ProposerVerdict(str(model_verdict).strip().lower())
    except ValueError:
        if confidence >= LINK_MIN_CONFIDENCE:
            verdict = ProposerVerdict.LINK
        elif confidence <= NO_LINK_MAX_CONFIDENCE:
            verdict = ProposerVerdict.NO_LINK
        else:
            verdict = ProposerVerdict.UNSURE

    if verdict == ProposerVerdict.LINK and n_grounded < MIN_GROUNDED_EVIDENCE_FOR_LINK:
        verdict = ProposerVerdict.UNSURE
    return verdict


# ---------------------------------------------------------------------------
# DECISÃO 2 — evidence must be quotable from the text the model was shown
# ---------------------------------------------------------------------------
GROUNDING_POLICY = "drop"  # "drop": remove unverified evidence | "flag": keep it, marked grounded=False
MIN_QUOTE_CHARS = 4        # shorter quotes ("a", "ok") match almost anything, so they do not count
# "temporal" evidence is about posting hours, not words, so there is no
# quote to look for. True: keep it without checking (it does not count as
# verified for the LINK rule). False: treat it like any other evidence.
TEMPORAL_EXEMPT_FROM_QUOTE_CHECK = True

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def is_grounded(
    quote_dark: str, quote_open: str, dark_sample: str, open_sample: str
) -> bool:
    """True if every quote the model gave appears verbatim (ignoring case
    and whitespace) in the sample of the alias it was attributed to, and
    at least one usable quote was given.
    """
    checks = [
        (q, s)
        for q, s in ((quote_dark, dark_sample), (quote_open, open_sample))
        if q and len(q.strip()) >= MIN_QUOTE_CHARS
    ]
    if not checks:
        return False
    return all(_norm(q) in _norm(s) for q, s in checks)


# ---------------------------------------------------------------------------
# DECISÃO 3 — what the model gets to see
# ---------------------------------------------------------------------------
SAMPLE_STRATEGY = "random"  # "first" | "random" | "longest"
SAMPLE_CHARS = 2000         # per alias
SAMPLE_SEED = 13            # a given alias always yields the same sample


def sample_messages(
    alias: Alias,
    budget_chars: int = SAMPLE_CHARS,
    strategy: str = SAMPLE_STRATEGY,
    seed: int = SAMPLE_SEED,
) -> str:
    """Pick whole messages until the character budget is used, and return
    them in their original order. The last message is cut to fit.
    """
    n = len(alias.messages)
    order = list(range(n))
    if strategy == "random":
        random.Random(f"{seed}:{alias.alias_id}").shuffle(order)
    elif strategy == "longest":
        order.sort(key=lambda i: len(alias.messages[i]), reverse=True)
    elif strategy != "first":
        raise ValueError(f"Unknown sample strategy {strategy!r}")

    chosen: list[tuple[int, str]] = []
    used = 0
    for i in order:
        remaining = budget_chars - used
        if remaining <= 0:
            break
        msg = alias.messages[i].strip()
        if not msg:
            continue
        msg = msg[:remaining]
        chosen.append((i, msg))
        used += len(msg) + 1  # +1 for the newline between messages
    chosen.sort(key=lambda t: t[0])
    return "\n".join(m for _, m in chosen)


_HOUR_BLOCKS = [(0, 5), (6, 11), (12, 17), (18, 23)]


def activity_summary(alias: Alias) -> str:
    """Share of posts per 6-hour block of the UTC day. The baseline uses
    the full hourly profile as a feature; the Proposer gets this coarse
    version so it can cite timing without being handed 24 numbers.
    """
    hours = alias.post_hours_utc
    if not hours:
        return "(no timestamps available)"
    total = len(hours)
    parts = []
    for lo, hi in _HOUR_BLOCKS:
        share = sum(1 for h in hours if lo <= h <= hi) / total
        parts.append(f"{lo:02d}-{hi:02d}h: {share:.0%}")
    return f"{total} posts; share by UTC hour block — " + ", ".join(parts)


# ---------------------------------------------------------------------------
# DECISÃO 4 — unusable replies
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 2  # first try + one retry with a corrective note

RETRY_NOTE = (
    "\n\nYour previous reply could not be used. Reply again with ONLY the JSON "
    "object described above: the keys verdict, confidence, hypothesis and "
    "evidence, with verdict one of link / no_link / unsure and confidence a "
    "number between 0 and 1."
)


SYSTEM_PROMPT = """\
You are the Proposer agent in a forensic identity-linking pipeline. You \
are given two text samples from two online aliases — one from a dark web \
forum/market, one from the open web — plus the aliases' posting-time \
profile and a stylometric similarity score from a baseline classifier. \
Your job is to decide whether they could be the same person, and to say \
exactly WHY.

Every piece of evidence MUST quote the text. Copy the exact words from the \
samples, character for character, into quote_dark and quote_open. Evidence \
without a quote that can be found in the samples will be thrown away, so do \
not paraphrase and do not invent. If you cannot quote anything that \
supports a link, say "unsure" or "no_link".

Be concrete. "The writing style is similar" is not useful; two quoted \
sentences that both open with "tbh" and discuss the same niche topic are \
useful, because they give the next agent (the Contestador) something \
specific to try to refute. Stated personal facts (city, age, job, habits) \
that match are the strongest kind of evidence; report them as \
"factual_support".

Verdicts:
- "link": the aliases are probably the same person.
- "no_link": the aliases are probably different people.
- "unsure": there is not enough to call it either way.

Respond with a single JSON object:
{
  "verdict": "link" | "no_link" | "unsure",
  "confidence": <float 0..1, your confidence that they ARE the same person>,
  "hypothesis": "<one or two sentences stating your call>",
  "evidence": [
    {"kind": "stylometric" | "factual_support" | "temporal",
     "description": "<what the quotes show>",
     "quote_dark": "<exact words from the dark web sample, or empty>",
     "quote_open": "<exact words from the open web sample, or empty>",
     "weight": <float 0..1>}
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
{score_line}
Dark web alias "{dark_id}"
Posting activity: {dark_activity}
Sample of their messages:
---
{dark_sample}
---

Open web alias "{open_id}"
Posting activity: {open_activity}
Sample of their messages:
---
{open_sample}
---

Decide whether these are the same person.
"""

_ALLOWED_KINDS = {"stylometric", "factual_support", "temporal"}


def _clamp01(value: Any, default: float) -> tuple[float, bool]:
    """Returns (value, ok). ok is False when `value` was not a usable number."""
    try:
        return max(0.0, min(1.0, float(value))), True
    except (TypeError, ValueError):
        return default, False


@dataclass
class ProposerAgent:
    llm: LLMClient
    sample_chars: int = SAMPLE_CHARS
    sample_strategy: str = SAMPLE_STRATEGY
    sample_seed: int = SAMPLE_SEED
    # Set False to run the "without the baseline score" ablation: does
    # showing the score anchor the model's call?
    show_baseline_score: bool = True
    max_attempts: int = MAX_ATTEMPTS

    def run(self, pair: CandidatePair, threshold: float = 0.4190) -> ProposerOutput:
        dark, open_ = pair.dark_alias, pair.open_alias
        dark_sample = sample_messages(dark, self.sample_chars, self.sample_strategy, self.sample_seed)
        open_sample = sample_messages(open_, self.sample_chars, self.sample_strategy, self.sample_seed)

        if self.show_baseline_score and pair.similarity_score is not None:
            score_line = (
                f"Baseline stylometric similarity score: {pair.similarity_score:.4f} "
                f"(threshold for the baseline's own accept decision: {threshold:.4f})"
            )
        else:
            score_line = "Baseline stylometric similarity score: not provided."

        user_prompt = USER_PROMPT_TEMPLATE.format(
            score_line=score_line,
            dark_id=dark.alias_id,
            dark_activity=activity_summary(dark),
            dark_sample=dark_sample,
            open_id=open_.alias_id,
            open_activity=activity_summary(open_),
            open_sample=open_sample,
        )

        raw = ""
        parsed: Optional[dict] = None
        last_error = ""
        for attempt in range(max(1, self.max_attempts)):
            prompt = user_prompt if attempt == 0 else user_prompt + RETRY_NOTE
            raw = self.llm.complete(SYSTEM_PROMPT, prompt, json_mode=True)
            try:
                candidate = safe_json_parse(raw)
            except ValueError as exc:  # includes json.JSONDecodeError
                last_error = f"attempt {attempt + 1}: {exc}"
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
            last_error = f"attempt {attempt + 1}: reply was not a JSON object"

        if parsed is None:
            return self._failed_output(pair, raw, last_error)
        return self._build_output(pair, parsed, raw, dark_sample, open_sample)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _baseline_evidence(pair: CandidatePair) -> Evidence:
        # The baseline score is always logged as evidence, so the Juiz can
        # see exactly what the single-agent baseline would have decided,
        # alongside what the Proposer adds on top of it.
        score = pair.similarity_score
        return Evidence(
            kind="stylometric",
            description=(
                "Baseline cosine similarity = n/a"
                if score is None
                else f"Baseline cosine similarity = {score:.4f}"
            ),
            weight=score,
            source_agent="baseline",
            raw={"similarity_score": score},
        )

    # DECISÃO 4 — what a failed call looks like downstream
    def _failed_output(self, pair: CandidatePair, raw: str, error: str) -> ProposerOutput:
        return ProposerOutput(
            pair=pair,
            hypothesis="",
            evidence=[self._baseline_evidence(pair)],
            confidence=0.0,
            verdict=ProposerVerdict.UNSURE,
            parse_error=True,
            raw_response=raw,
            warnings=[f"unusable reply after {self.max_attempts} attempt(s): {error}"],
        )

    def _build_output(
        self,
        pair: CandidatePair,
        parsed: dict,
        raw: str,
        dark_sample: str,
        open_sample: str,
    ) -> ProposerOutput:
        warnings: list[str] = []

        confidence, ok = _clamp01(parsed.get("confidence"), default=0.5)
        if not ok:
            warnings.append(f"confidence {parsed.get('confidence')!r} is not a number; using 0.5")

        kept: list[Evidence] = []
        dropped: list[dict[str, Any]] = []
        n_grounded = 0
        items = parsed.get("evidence")
        if not isinstance(items, list):
            if items is not None:
                warnings.append("evidence was not a list; ignored")
            items = []

        for item in items:
            if not isinstance(item, dict) or not str(item.get("description", "")).strip():
                warnings.append("evidence item without a description; skipped")
                continue
            kind = str(item.get("kind", "stylometric")).strip().lower()
            if kind not in _ALLOWED_KINDS:
                kind = "stylometric"
            weight, _ = _clamp01(item.get("weight"), default=0.5)
            quote_dark = str(item.get("quote_dark") or "")
            quote_open = str(item.get("quote_open") or "")
            exempt = kind == "temporal" and TEMPORAL_EXEMPT_FROM_QUOTE_CHECK
            grounded = is_grounded(quote_dark, quote_open, dark_sample, open_sample)

            record = {
                "kind": kind,
                "description": str(item["description"]).strip(),
                "weight": weight,
                "quote_dark": quote_dark,
                "quote_open": quote_open,
                "grounded": grounded,
            }
            if grounded:
                n_grounded += 1
            elif GROUNDING_POLICY == "drop" and not exempt:
                dropped.append(record)
                continue
            kept.append(
                Evidence(
                    kind=kind,
                    description=record["description"],
                    weight=weight,
                    source_agent="proposer",
                    raw={k: record[k] for k in ("quote_dark", "quote_open", "grounded")},
                )
            )

        verdict = decide_verdict(parsed.get("verdict"), confidence, n_grounded)

        return ProposerOutput(
            pair=pair,
            hypothesis=str(parsed.get("hypothesis", "")).strip(),
            evidence=[self._baseline_evidence(pair)] + kept,
            confidence=confidence,
            verdict=verdict,
            raw_response=raw,
            dropped_evidence=dropped,
            warnings=warnings,
        )
