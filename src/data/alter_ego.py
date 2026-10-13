"""Synthetic ground truth via alter-ego splitting.

Reproduces the validation methodology from Arabnezhad et al. (2020),
Section V.A: take a real user's messages, randomly split them into two
disjoint halves, and treat each half as a separate alias. The pipeline is
then asked to re-match the two halves. Because the "true" pairing is known
by construction, this gives precision/recall numbers without needing any
labeled real-world matches.

Important: this does NOT require the paper's own (non-public) dataset. It
only requires *a* corpus of users with enough text/timestamps per user —
e.g. a public Reddit pull, or Gwern Branwen's archived dark-web forum
dumps (used by Ho & Ng, 2016; see the TCC's doc, "Metodologia e Dados").
Point `build_alter_ego_pairs` at whatever corpus you have loaded into
`Alias` objects.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.data.features import MIN_TIMESTAMPS, MIN_WORDS, meets_minimum_activity
from src.schemas import Alias, Platform


@dataclass
class AlterEgoPair:
    original: Alias
    alter_ego: Alias
    source_alias_id: str  # the real alias both halves were split from


def _split_two(items: list, rng: random.Random) -> tuple[list, list]:
    shuffled = items[:]
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def build_alter_ego_pairs(
    aliases: list[Alias],
    platform_label: Platform = Platform.OPEN_WEB,
    seed: int = 13,
    require_minimums_on_each_half: bool = True,
) -> list[AlterEgoPair]:
    """For every alias with enough text, split its messages (and matching
    timestamps) in two and return both halves as an AlterEgoPair.

    `require_minimums_on_each_half`: the paper requires >=1500 words and
    >=30 timestamps on the *combined* user before splitting (Table III
    sweeps this threshold). Set this True to additionally require each
    half to individually clear the bar — a stricter, smaller, cleaner
    benchmark; False to match the paper's own protocol more closely.
    """
    rng = random.Random(seed)
    pairs: list[AlterEgoPair] = []

    for alias in aliases:
        if not meets_minimum_activity(alias):
            continue

        # messages and post_hours_utc are assumed aligned by index
        if len(alias.messages) != len(alias.post_hours_utc):
            raise ValueError(
                f"{alias.alias_id}: messages ({len(alias.messages)}) and "
                f"post_hours_utc ({len(alias.post_hours_utc)}) must be the same length "
                "so a split keeps each message with its own timestamp."
            )

        indices = list(range(len(alias.messages)))
        idx_a, idx_b = _split_two(indices, rng)
        if not idx_a or not idx_b:
            continue

        original = Alias(
            alias_id=f"{alias.alias_id}__orig",
            platform=platform_label,
            source=alias.source,
            messages=[alias.messages[i] for i in idx_a],
            post_hours_utc=[alias.post_hours_utc[i] for i in idx_a],
        )
        alter_ego = Alias(
            alias_id=f"{alias.alias_id}__alter",
            platform=platform_label,
            source=alias.source,
            messages=[alias.messages[i] for i in idx_b],
            post_hours_utc=[alias.post_hours_utc[i] for i in idx_b],
        )

        if require_minimums_on_each_half:
            if not (meets_minimum_activity(original) and meets_minimum_activity(alter_ego)):
                continue

        pairs.append(AlterEgoPair(original=original, alter_ego=alter_ego, source_alias_id=alias.alias_id))

    return pairs


def negative_pairs(
    alter_ego_pairs: list[AlterEgoPair], n_per_positive: int = 1, seed: int = 13
) -> list[tuple[Alias, Alias]]:
    """Mismatched (negative) pairs for evaluation: an `original` half
    paired with a *different* user's `alter_ego` half. Needed alongside
    the true positives above so precision/recall are measurable rather
    than trivially 100% recall on an all-positive set.
    """
    rng = random.Random(seed + 1)
    negatives: list[tuple[Alias, Alias]] = []
    n = len(alter_ego_pairs)
    if n < 2:
        return negatives

    for i, pair in enumerate(alter_ego_pairs):
        others = [j for j in range(n) if j != i]
        for j in rng.sample(others, k=min(n_per_positive, len(others))):
            negatives.append((pair.original, alter_ego_pairs[j].alter_ego))

    return negatives
