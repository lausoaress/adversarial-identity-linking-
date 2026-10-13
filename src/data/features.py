"""Stylometric and temporal feature extraction.

This reproduces the exact feature set from the baseline paper (Arabnezhad
et al., 2020, "A Light in the Dark Web", Section IV):

  - word n-grams (n = 1..3)
  - character n-grams (n = 1..5)
  - punctuation frequency ('.', ',', ':')
  - digit frequency ('0'-'9')
  - special-character frequency ('@', '#', ...)
  - a daily activity-time "fingerprint":

        P_u[h] = sum_d a_u(d, h) / sum_{d, h'} a_u(d, h')

    i.e. for user u, the fraction of their posts (across all collected
    days d) that fall in UTC hour h. This is what the baseline uses
    *in addition to* stylometry, and it is one of the two signal
    families the Proposer draws on.

Reproducing this exactly (rather than inventing a new feature set) matters
for the TCC: it lets you plug the adversarial layer on top of the same
evidence the baseline sees, so any gain is attributable to the
verification architecture and not to better features.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import Alias

# Minimums from the paper: a user needs at least this much signal before
# their features are considered reliable (Section IV.B / V.A).
MIN_WORDS = 1500
MIN_TIMESTAMPS = 30

PUNCTUATION_CHARS = [".", ",", ":"]
DIGIT_CHARS = list("0123456789")
SPECIAL_CHARS = list("@#!?;()[]{}\"'-_/\\&%$+=<>~`|^*")


@dataclass
class StylometricVectorizers:
    """Holds the fitted vectorizers so new aliases can be transformed into
    the *same* feature space as the ones a vectorizer was fit on (cosine
    similarity is only meaningful in a shared space).
    """

    word_ngrams: CountVectorizer
    char_ngrams: CountVectorizer


def meets_minimum_activity(alias: Alias) -> bool:
    """The paper excludes users below the word/timestamp floor — doing the
    same here keeps the prototype's numbers comparable to the baseline's.
    """
    return alias.word_count >= MIN_WORDS and len(alias.post_hours_utc) >= MIN_TIMESTAMPS


def fit_vectorizers(
    corpus_texts: list[str],
    word_ngram_features: int = 50_000,
    char_ngram_features: int = 15_000,
) -> StylometricVectorizers:
    """Fit word (1-3) and char (1-5) n-gram vectorizers over a corpus.

    `word_ngram_features` / `char_ngram_features` default to the "final
    classification" sizes from the paper (Table II); pass the larger
    "space reduction" sizes (60k / 30k) if you are replicating the
    two-stage k-attribution step instead (see src/baseline/cosine_pipeline.py).
    """
    word_vec = CountVectorizer(
        ngram_range=(1, 3), max_features=word_ngram_features, analyzer="word"
    )
    char_vec = CountVectorizer(
        ngram_range=(1, 5), max_features=char_ngram_features, analyzer="char"
    )
    word_vec.fit(corpus_texts)
    char_vec.fit(corpus_texts)
    return StylometricVectorizers(word_ngrams=word_vec, char_ngrams=char_vec)


def _char_frequency_vector(text: str, chars: list[str]) -> np.ndarray:
    length = max(len(text), 1)
    counts = Counter(text)
    return np.array([counts.get(c, 0) / length for c in chars], dtype=np.float64)


def stylometric_vector(alias: Alias, vectorizers: StylometricVectorizers) -> csr_matrix:
    """The full stylometric feature vector for one alias: word n-grams +
    char n-grams + punctuation/digit/special-char frequencies, exactly the
    four families listed in the paper.
    """
    text = alias.text
    word_feats = vectorizers.word_ngrams.transform([text])
    char_feats = vectorizers.char_ngrams.transform([text])
    punct_feats = csr_matrix(_char_frequency_vector(text, PUNCTUATION_CHARS))
    digit_feats = csr_matrix(_char_frequency_vector(text, DIGIT_CHARS))
    special_feats = csr_matrix(_char_frequency_vector(text, SPECIAL_CHARS))
    return hstack([word_feats, char_feats, punct_feats, digit_feats, special_feats]).tocsr()


def activity_fingerprint(alias: Alias) -> np.ndarray:
    """P_u[h] for h in 0..23 — the fraction of this alias's posts made in
    each UTC hour. Weekends/holidays are expected to already be filtered
    out of `post_hours_utc` upstream (the paper excludes them when
    building the profile); this function just does the division.
    """
    hours = np.zeros(24, dtype=np.float64)
    for h in alias.post_hours_utc:
        hours[h % 24] += 1
    total = hours.sum()
    if total == 0:
        return hours  # all-zero vector; caller should treat this alias as unreliable
    return hours / total


def combined_similarity(
    a: Alias,
    b: Alias,
    vectorizers: StylometricVectorizers,
    stylometric_weight: float = 0.7,
    temporal_weight: float = 0.3,
) -> float:
    """Cosine similarity over stylometric features, blended with cosine
    similarity over the activity fingerprint. The paper's own pipeline
    uses stylometric cosine similarity for the actual matching decision;
    the temporal fingerprint is additional corroborating evidence. The
    weights are a starting point, not a finding — tune them once you have
    labeled pairs to validate against (see src/evaluation/metrics.py).
    """
    style_a = stylometric_vector(a, vectorizers)
    style_b = stylometric_vector(b, vectorizers)
    style_sim = float(cosine_similarity(style_a, style_b)[0, 0])

    temp_a = activity_fingerprint(a).reshape(1, -1)
    temp_b = activity_fingerprint(b).reshape(1, -1)
    if temp_a.sum() == 0 or temp_b.sum() == 0:
        temp_sim = 0.0
    else:
        temp_sim = float(cosine_similarity(temp_a, temp_b)[0, 0])

    return stylometric_weight * style_sim + temporal_weight * temp_sim


_WORD_RE = re.compile(r"\b\w+\b")


def quick_stats(alias: Alias) -> dict:
    """Cheap, vectorizer-free sanity numbers — useful when eyeballing a
    newly loaded corpus before committing to a full feature-extraction run.
    """
    text = alias.text
    return {
        "alias_id": alias.alias_id,
        "n_messages": len(alias.messages),
        "word_count": alias.word_count,
        "n_timestamps": len(alias.post_hours_utc),
        "meets_minimum": meets_minimum_activity(alias),
        "punct_density": sum(text.count(c) for c in PUNCTUATION_CHARS) / max(len(text), 1),
    }
