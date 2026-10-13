"""Corpus loaders.

Three real loaders, plus the synthetic generator used everywhere else in
development:

1. `load_reddit_csv` — reads a one-row-per-post CSV, the shape Pushshift
   dumps and Academic Torrents exports commonly use (author, body,
   created_utc as a Unix epoch). Works today against that format.

2. `load_reddit_api` — pulls a specific list of usernames' own posts via
   the official Reddit API (PRAW). Needs a registered Reddit "script" app
   (https://www.reddit.com/prefs/apps) and REDDIT_CLIENT_ID /
   REDDIT_CLIENT_SECRET (see .env.example). `praw` is an optional
   dependency — only imported if you call this function.

3. `load_forum_dump` — reads an extracted wget-mirrored HTML archive (the
   format Gwern Branwen's dark-web archive uses, including
   `themajesticgarden-forums.tar.xz` — see data/README.md). The CSS
   selectors in `ForumDumpSelectors` are generic phpBB-style defaults:
   extract one real page from your downloaded archive and adjust them to
   match before trusting the output on the full corpus. `beautifulsoup4`
   is a dependency of this loader only.

`make_synthetic_corpus` has no external dependency, so every other module
(features, alter-ego splitting, the baseline, the agents) can be
developed and tested without any of the above.
"""
from __future__ import annotations

import csv
import itertools
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import Alias, Platform

# ---------------------------------------------------------------------------
# Reddit — CSV export (Pushshift / Academic Torrents style dumps)
# ---------------------------------------------------------------------------


def load_reddit_csv(
    path: str,
    source: str = "reddit",
    author_col: str = "author",
    body_col: str = "body",
    timestamp_col: str = "created_utc",
    timestamp_is_epoch: bool = True,
) -> list[Alias]:
    """Reads a CSV with one row per post.

    Default column names match Pushshift/Academic-Torrents-style Reddit
    dumps: `author`, `body`, `created_utc` (Unix epoch seconds). If your
    export already has a precomputed UTC hour column instead of an epoch
    timestamp, pass `timestamp_col="created_utc_hour"` and
    `timestamp_is_epoch=False`.

    Rows with a missing/unparseable timestamp are kept (message counted
    for stylometric features) but contribute no hour to the activity
    fingerprint, rather than dropping the whole row — a few bad rows in a
    real scrape shouldn't silently shrink your corpus.
    """
    users: dict[str, Alias] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author = row.get(author_col)
            body = row.get(body_col)
            if not author or not body:
                continue
            if author not in users:
                users[author] = Alias(alias_id=author, platform=Platform.OPEN_WEB, source=source)
            users[author].messages.append(body)

            raw_ts = row.get(timestamp_col)
            if raw_ts:
                try:
                    if timestamp_is_epoch:
                        hour = datetime.fromtimestamp(int(float(raw_ts)), tz=timezone.utc).hour
                    else:
                        hour = int(raw_ts)
                    users[author].post_hours_utc.append(hour)
                except (ValueError, OverflowError, OSError):
                    pass  # unparseable timestamp - message still counted above
    return list(users.values())


# ---------------------------------------------------------------------------
# Reddit — official API (PRAW), for a known list of usernames
# ---------------------------------------------------------------------------


def load_reddit_api(
    usernames: list[str],
    source: str = "reddit",
    limit_per_user: int = 500,
    client_id: str | None = None,
    client_secret: str | None = None,
    user_agent: str = "adversarial-identity-linking/0.1 (UFSC undergraduate thesis)",
) -> list[Alias]:
    """Pulls each username's own submissions + comments via the official
    Reddit API.

    Needs a registered Reddit "script" app (free, instant approval):
    https://www.reddit.com/prefs/apps -> "create another app" -> type
    "script". Reads `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` from the
    environment if not passed explicitly (see .env.example) - this
    function never hardcodes credentials.

    A suspended, shadowbanned, or private account raises inside PRAW; we
    skip that user rather than aborting the whole pull, since a handful
    of dead accounts is normal in any real username list.
    """
    import praw  # optional dependency - only needed if you call this loader

    client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(
            "Missing Reddit API credentials. Set REDDIT_CLIENT_ID and "
            "REDDIT_CLIENT_SECRET (e.g. in a local .env you load before "
            "running), or pass client_id=/client_secret= directly. "
            "Register a free app at https://www.reddit.com/prefs/apps."
        )

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    aliases: list[Alias] = []
    for username in usernames:
        messages: list[str] = []
        hours: list[int] = []
        try:
            redditor = reddit.redditor(username)
            items = itertools.chain(
                redditor.submissions.new(limit=limit_per_user),
                redditor.comments.new(limit=limit_per_user),
            )
            for item in items:
                text = getattr(item, "selftext", "") or getattr(item, "body", "")
                if not text or text in ("[deleted]", "[removed]"):
                    continue
                messages.append(text)
                hours.append(datetime.fromtimestamp(item.created_utc, tz=timezone.utc).hour)
        except Exception:
            # Suspended / shadowbanned / private / deleted account, or a
            # transient API error for this one user - skip, don't abort
            # the whole pull over one bad username.
            continue

        if messages:
            aliases.append(
                Alias(
                    alias_id=username,
                    platform=Platform.OPEN_WEB,
                    source=source,
                    messages=messages,
                    post_hours_utc=hours,
                )
            )
    return aliases


# ---------------------------------------------------------------------------
# Dark-web forum — extracted wget-mirrored HTML archive (Gwern's dnm-archive)
# ---------------------------------------------------------------------------


@dataclass
class ForumDumpSelectors:
    """CSS selectors locating one post's pieces within a mirrored HTML
    page. The defaults below are generic phpBB-style guesses - Gwern's
    archive page (https://gwern.net/dnm-archive) confirms each forum's
    files are plain wget crawls with the site's own HTML, but does not
    document The Majestic Garden's specific markup. Once you've extracted
    `themajesticgarden-forums.tar.xz`, open one real thread page, check
    these selectors against it in a browser dev console, and override
    whichever don't match via config/config.yaml - see data/README.md.
    """

    post: str = ".post, .postbody, article"
    author: str = ".post-author, .username, .author"
    timestamp: str = "time, .post-date, .postdate"
    timestamp_attr: str | None = "datetime"  # read this attribute first; fall back to the tag's text
    body: str = ".post-body, .postbody, .post-content, .content"


_TIME_IN_TEXT = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")


def _extract_hour(tag, attr: str | None) -> int | None:
    """Best-effort hour extraction: try the given attribute as an ISO-8601
    timestamp first (what `datetime` HTML elements normally carry), then
    fall back to spotting an HH:MM pattern in the visible text. Returns
    None rather than raising - a post we can't date is still kept for
    stylometric features, just excluded from the activity fingerprint.
    """
    if tag is None:
        return None
    if attr:
        raw = tag.get(attr)
        if raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).hour
            except ValueError:
                pass
    text = tag.get_text(" ", strip=True)
    match = _TIME_IN_TEXT.search(text)
    if match:
        return int(match.group(1))
    return None


def load_forum_dump(
    root_dir: str,
    source: str = "dark_web_forum",
    selectors: ForumDumpSelectors | None = None,
    file_glob: str = "**/*.html",
) -> list[Alias]:
    """Walks `root_dir` (the directory you extracted an archive like
    `themajesticgarden-forums.tar.xz` into - date-stamped subfolders of a
    plain wget crawl, per Gwern's archive page), parses every matching
    HTML file, and groups posts by author into `Alias` objects.

    Raises `ImportError` with an install hint if `beautifulsoup4` isn't
    installed - it's a dependency of this loader only, not the whole
    project. A page with zero elements matching `selectors.post` is
    skipped rather than raising (mirrors routinely include index/nav
    pages with no posts on them), so a wrong selector shows up as "0
    users found", not a crash - inspect one real file's structure if that
    happens.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "load_forum_dump needs beautifulsoup4: pip install beautifulsoup4"
        ) from exc

    selectors = selectors or ForumDumpSelectors()
    users: dict[str, Alias] = {}

    for html_path in Path(root_dir).glob(file_glob):
        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        soup = BeautifulSoup(html, "html.parser")

        for post in soup.select(selectors.post):
            author_tag = post.select_one(selectors.author)
            body_tag = post.select_one(selectors.body)
            if author_tag is None or body_tag is None:
                continue
            author = author_tag.get_text(strip=True)
            body = body_tag.get_text(" ", strip=True)
            if not author or not body:
                continue

            if author not in users:
                users[author] = Alias(alias_id=author, platform=Platform.DARK_WEB, source=source)
            users[author].messages.append(body)

            hour = _extract_hour(post.select_one(selectors.timestamp), selectors.timestamp_attr)
            if hour is not None:
                users[author].post_hours_utc.append(hour)

    return list(users.values())


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
