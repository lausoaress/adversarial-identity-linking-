"""Core data structures shared across the pipeline.

Keeping these as plain dataclasses (no framework coupling) means the
baseline pipeline, the agents, and the evaluation code can all speak the
same language without importing each other's internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Platform(str, Enum):
    DARK_WEB = "dark_web"
    OPEN_WEB = "open_web"


@dataclass
class Alias:
    """One identity on one platform: a username plus the raw messages
    and timestamps used to build evidence about them.
    """

    alias_id: str
    platform: Platform
    source: str  # e.g. "dream_market", "the_majestic_garden", "reddit"
    messages: list[str] = field(default_factory=list)
    # UTC hour-of-day for each post, used for the activity fingerprint
    # (Arabnezhad et al., 2020 — Section IV.B).
    post_hours_utc: list[int] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(m.split()) for m in self.messages)

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


@dataclass
class Evidence:
    """One piece of support (or contradiction) for a candidate pair.

    `kind` distinguishes the stylometric/behavioral evidence the baseline
    already produces from the factual-contradiction evidence that is the
    Contestador's novel contribution (see doc section "Trabalhos
    Relacionados" — the 43% "Unclear" finding in Arabnezhad et al.).
    """

    kind: str  # "stylometric" | "temporal" | "factual_contradiction" | "factual_support"
    description: str
    weight: Optional[float] = None
    source_agent: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePair:
    """A proposed link between a dark-web alias and an open-web alias."""

    dark_alias: Alias
    open_alias: Alias
    similarity_score: Optional[float] = None  # from the baseline cosine stage


class ProposerVerdict(str, Enum):
    """The Proposer's own call on a pair. Distinct from the Juiz's
    `Verdict` (accept/reject/unclear): this is the single-LLM-agent
    decision, which is also the control arm of the experiment
    (baseline vs. Proposer alone vs. full adversarial pipeline).
    """

    LINK = "link"        # likely the same person
    NO_LINK = "no_link"  # likely different people
    UNSURE = "unsure"    # not enough to call it either way


@dataclass
class ProposerOutput:
    pair: CandidatePair
    hypothesis: str
    evidence: list[Evidence]
    confidence: float  # 0..1, Proposer's own estimate that the aliases are the same person
    verdict: ProposerVerdict = ProposerVerdict.UNSURE
    # True when the model's reply could not be parsed even after retrying.
    # Such a result is a placeholder, not a judgement: evaluation code should
    # count these separately instead of treating them as a real UNSURE.
    parse_error: bool = False
    raw_response: str = ""  # last raw model reply, kept for auditing
    # Evidence the model reported but that failed the quote check (see
    # src/agents/proposer.py). Kept for analysis, not shown to later agents.
    dropped_evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ContestadorOutput:
    pair: CandidatePair
    rebuttal: str
    evidence: list[Evidence]
    stance: str  # "supports" | "contests" | "inconclusive"


class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNCLEAR = "unclear"  # the Juiz may still emit this — the goal is to shrink, not hide, this bucket


@dataclass
class JuizOutput:
    pair: CandidatePair
    verdict: Verdict
    explanation: str
    cited_evidence: list[Evidence]
    rounds_used: int = 1


@dataclass
class PipelineResult:
    """What one candidate pair looks like after going through the full
    Proposer -> Contestador -> Juiz pipeline. This is the unit the
    evaluation code compares against the baseline and, where available,
    ground truth.
    """

    pair: CandidatePair
    proposer: ProposerOutput
    contestador: ContestadorOutput
    juiz: JuizOutput
    ground_truth: Optional[bool] = None  # True = same person, False = different, None = unknown
