"""Corpus loaders.

Two kinds of loader live here:

1. Real loaders (`load_reddit_csv`, `load_forum_dump`) — thin, deliberately
   simple readers you fill in once you've pulled an actual corpus. They
   are stubs on purpose: the right format depends on which public source
   you end up using (Reddit API/pushshift export vs. Gwern's dark-web
   archive layout), which is a data-access decision, not a code one.

2. `make_synthetic_corpus` — a fake-but-structurally-realistic corpus
   generator with no external dependency, so every other module (features,
   alter-ego splitting, the baseline, the agents) can be developed and
   tested *today*, before real data access is sorted out.
"""
from __future__ import annotations

import csv
import random

from src.schemas import Alias, Platform

# ---------------------------------------------------------------------------
# Real loaders — fill in once a corpus is on disk.
# ---------------------------------------------------------------------------


def load_reddit_csv(path: str, source: str = "reddit") -> list[Alias]:
    """Expects a CSV with columns: author, body, created_utc_hour.
    One row per post. Adjust to match whatever export format you pull
    (Reddit API, Pushshift dump, Academic Torrents, etc.).
    """
    users: dict[str, Alias] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author = row["author"]
            if author not in users:
                users[author] = Alias(
                    alias_id=author, platform=Platform.OPEN_WEB, source=source
                )
            users[author].messages.append(row["body"])
            users[author].post_hours_utc.append(int(row["created_utc_hour"]))
    return list(users.values())


def load_forum_dump(path: str, source: str = "dark_web_forum") -> list[Alias]:
    """Placeholder for Gwern Branwen's dark-web forum archive format (or
    any similarly structured dump). Implement once you've picked a
    specific archive and inspected its layout.
    """
    raise NotImplementedError(
        "Point this at the specific archive format you're using "
        "(see the TCC doc's Metodologia e Dados section for candidate sources)."
    )


# ---------------------------------------------------------------------------
# Synthetic corpus — for developing and smoke-testing everything else.
# ---------------------------------------------------------------------------

_TOPICS = [
    "market", "vendor", "escrow", "shipping", "pgp key", "review", "price",
    "encryption", "scam", "moderator", "forum rules", "login", "wallet",
]
_FILLER = [
    "honestly", "i think", "in my experience", "to be fair", "as always",
    "no cap", "imo", "tbh", "for real", "just saying",
]


def _fake_message(rng: random.Random, vocabulary: list[str]) -> str:
    length = rng.randint(8, 24)
    words = [rng.choice(vocabulary) for _ in range(length)]
    if rng.random() < 0.3:
        words.insert(0, rng.choice(_FILLER))
    punctuation = rng.choice([".", "...", "!", "?", ""])
    return " ".join(words) + punctuation


def make_synthetic_corpus(
    n_users: int = 20,
    messages_per_user: tuple[int, int] = (120, 220),
    platform: Platform = Platform.OPEN_WEB,
    source: str = "synthetic",
    seed: int = 7,
    active_hours: tuple[int, int] = (8, 23),
) -> list[Alias]:
    """Generates `n_users` aliases, each with a distinct vocabulary bias
    (so stylometric similarity is non-trivial) and a distinct preferred
    posting-hour window (so the temporal fingerprint is non-trivial too).

    This is intentionally crude — it exists to let the pipeline run end to
    end, not to stand in for a real evaluation.
    """
    rng = random.Random(seed)
    aliases: list[Alias] = []

    for u in range(n_users):
        # give each synthetic user a personal 4-6 word "signature" vocabulary
        # mixed into the shared topic pool, so some pairs are more similar
        # than others (rather than everything being uniformly random).
        signature = rng.sample(
            [f"term{k}" for k in range(50)], k=rng.randint(4, 6)
        )
        vocabulary = _TOPICS + signature

        n_msgs = rng.randint(*messages_per_user)
        hour_center = rng.randint(*active_hours)

        messages = [_fake_message(rng, vocabulary) for _ in range(n_msgs)]
        hours = [
            max(0, min(23, int(rng.gauss(hour_center, 2)))) for _ in range(n_msgs)
        ]

        aliases.append(
            Alias(
                alias_id=f"user_{u:03d}",
                platform=platform,
                source=source,
                messages=messages,
                post_hours_utc=hours,
            )
        )

    return aliases
