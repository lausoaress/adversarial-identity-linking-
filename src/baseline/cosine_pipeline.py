"""The single-agent baseline (Arabnezhad et al., 2020).

Two stages, exactly as described in the paper (Section IV):

  1. k-attribution: reduce a large candidate pool to the top-k most
     similar open-web aliases for each dark-web alias, using a cheap
     feature set (the "space reduction" sizes: 60k word n-grams, 30k char
     n-grams).
  2. Final matching: recompute features at the smaller "final
     classification" sizes (50k / 15k) on just those k candidates, and
     accept the pair if cosine similarity clears the paper's threshold
     (0.4190).

This is the thing the adversarial architecture (Proposer/Contestador/Juiz)
has to *beat* — specifically, it's the thing that leaves ~43% of
real-world candidate pairs "Unclear" rather than confidently matched
(see the TCC doc's "Trabalhos Relacionados" section). Keep this module
unchanged as you build the adversarial pipeline, so the two can be run
side by side on the same inputs (src/evaluation/run_experiment.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.data.features import (
    StylometricVectorizers,
    fit_vectorizers,
    meets_minimum_activity,
    stylometric_vector,
)
from src.schemas import Alias, CandidatePair

DEFAULT_THRESHOLD = 0.4190  # Arabnezhad et al., Section V.B
DEFAULT_K = 10

REDUCTION_WORD_FEATURES = 60_000
REDUCTION_CHAR_FEATURES = 30_000
FINAL_WORD_FEATURES = 50_000
FINAL_CHAR_FEATURES = 15_000


@dataclass
class BaselineConfig:
    k: int = DEFAULT_K
    threshold: float = DEFAULT_THRESHOLD


def k_attribution(
    dark_alias: Alias,
    open_candidates: list[Alias],
    config: BaselineConfig = BaselineConfig(),
) -> list[Alias]:
    """Stage 1: rank open-web candidates by cosine similarity on the
    "space reduction" feature sizes and keep the top k.
    """
    corpus = [dark_alias.text] + [c.text for c in open_candidates]
    vecs = fit_vectorizers(
        corpus,
        word_ngram_features=REDUCTION_WORD_FEATURES,
        char_ngram_features=REDUCTION_CHAR_FEATURES,
    )
    dark_vec = stylometric_vector(dark_alias, vecs)
    sims = [
        float(cosine_similarity(dark_vec, stylometric_vector(c, vecs))[0, 0])
        for c in open_candidates
    ]
    ranked = sorted(zip(open_candidates, sims), key=lambda t: t[1], reverse=True)
    return [alias for alias, _ in ranked[: config.k]]


def final_match(
    dark_alias: Alias,
    candidates: list[Alias],
    config: BaselineConfig = BaselineConfig(),
) -> list[CandidatePair]:
    """Stage 2: recompute at the final feature sizes over just the k
    survivors, and attach a similarity_score to each resulting pair. The
    caller decides what to do with pairs below `config.threshold` (the
    paper's own headline numbers are precision/recall AT this threshold,
    not a hard accept/reject — see src/evaluation/metrics.py).
    """
    if not candidates:
        return []
    corpus = [dark_alias.text] + [c.text for c in candidates]
    vecs = fit_vectorizers(
        corpus,
        word_ngram_features=FINAL_WORD_FEATURES,
        char_ngram_features=FINAL_CHAR_FEATURES,
    )
    dark_vec = stylometric_vector(dark_alias, vecs)
    pairs = []
    for c in candidates:
        sim = float(cosine_similarity(dark_vec, stylometric_vector(c, vecs))[0, 0])
        pairs.append(CandidatePair(dark_alias=dark_alias, open_alias=c, similarity_score=sim))
    return pairs


def run_baseline(
    dark_alias: Alias,
    open_candidates: list[Alias],
    config: BaselineConfig = BaselineConfig(),
) -> list[CandidatePair]:
    """The full two-stage pipeline for one dark-web alias against a pool
    of open-web candidates. Returns scored pairs for the surviving k
    candidates, sorted by similarity, highest first.
    """
    if not meets_minimum_activity(dark_alias):
        return []
    reliable_candidates = [c for c in open_candidates if meets_minimum_activity(c)]
    shortlisted = k_attribution(dark_alias, reliable_candidates, config)
    pairs = final_match(dark_alias, shortlisted, config)
    return sorted(pairs, key=lambda p: p.similarity_score or 0.0, reverse=True)


def accepts(pair: CandidatePair, config: BaselineConfig = BaselineConfig()) -> bool:
    """The baseline's binary decision: a single threshold, no second
    opinion. This is the exact decision rule the adversarial pipeline
    is meant to improve on.
    """
    return (pair.similarity_score or 0.0) >= config.threshold
