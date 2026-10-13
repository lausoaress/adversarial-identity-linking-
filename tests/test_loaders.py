"""Tests for the real (non-synthetic) loaders in src/data/loaders.py.

load_reddit_api isn't tested here - it needs praw (optional dependency)
and live API credentials, which this sandbox has neither of. Everything
testable without a real archive/API key is covered: the CSV parser's
epoch-timestamp handling, and the HTML forum-dump parser against a small
fixture page shaped like a generic phpBB-style mirror (see
ForumDumpSelectors' docstring on why this is a best guess, not a
confirmed match to The Majestic Garden's real markup).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.data.loaders import ForumDumpSelectors, load_forum_dump, load_reddit_csv

_FIXTURE_HTML = """
<html><body>
  <div class="post">
    <span class="post-author">alice</span>
    <time datetime="2017-05-03T14:22:00Z">May 3</time>
    <div class="post-body">loved this vendor's escrow process, fast shipping</div>
  </div>
  <div class="post">
    <span class="post-author">bob</span>
    <time datetime="2017-05-03T03:05:00Z">May 3</time>
    <div class="post-body">pgp key verification failed for me, anyone else?</div>
  </div>
  <div class="post">
    <span class="post-author">alice</span>
    <time datetime="2017-05-04T15:10:00Z">May 4</time>
    <div class="post-body">second post from alice, different day same hour-ish</div>
  </div>
  <nav class="site-nav">not a post, should be ignored by the .post selector</nav>
</body></html>
"""


def test_load_forum_dump_groups_posts_by_author(tmp_path: Path):
    page = tmp_path / "2017-05-03" / "thread1.html"
    page.parent.mkdir(parents=True)
    page.write_text(_FIXTURE_HTML, encoding="utf-8")

    aliases = load_forum_dump(str(tmp_path))
    by_id = {a.alias_id: a for a in aliases}

    assert set(by_id) == {"alice", "bob"}
    assert len(by_id["alice"].messages) == 2
    assert len(by_id["bob"].messages) == 1
    # datetime="...T14:22:00Z" -> hour 14, read from the `datetime` attribute
    assert by_id["alice"].post_hours_utc == [14, 15]
    assert by_id["bob"].post_hours_utc == [3]


def test_load_forum_dump_on_directory_with_no_matching_posts_returns_empty(tmp_path: Path):
    page = tmp_path / "empty.html"
    page.write_text("<html><body><p>nothing here</p></body></html>", encoding="utf-8")

    assert load_forum_dump(str(tmp_path)) == []


def test_load_forum_dump_custom_selectors(tmp_path: Path):
    # A differently-marked-up page - confirms selectors are actually
    # overridable per data/README.md's "inspect the real archive" step,
    # not hardcoded.
    page = tmp_path / "custom.html"
    page.write_text(
        '<article><p class="who">carol</p><p class="msg">hello world</p></article>',
        encoding="utf-8",
    )
    selectors = ForumDumpSelectors(post="article", author=".who", body=".msg", timestamp="time")
    aliases = load_forum_dump(str(tmp_path), selectors=selectors)

    assert len(aliases) == 1
    assert aliases[0].alias_id == "carol"
    assert aliases[0].messages == ["hello world"]
    assert aliases[0].post_hours_utc == []  # no <time> element on this fixture - fine, not fatal


def test_load_reddit_csv_parses_epoch_timestamp(tmp_path: Path):
    path = tmp_path / "reddit.csv"
    ts = int(datetime(2020, 1, 1, 9, 30, tzinfo=timezone.utc).timestamp())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["author", "body", "created_utc"])
        writer.writeheader()
        writer.writerow({"author": "dave", "body": "first post", "created_utc": str(ts)})
        writer.writerow({"author": "dave", "body": "second post", "created_utc": str(ts + 3600)})

    aliases = load_reddit_csv(str(path))
    assert len(aliases) == 1
    assert aliases[0].alias_id == "dave"
    assert aliases[0].messages == ["first post", "second post"]
    assert aliases[0].post_hours_utc == [9, 10]


def test_load_reddit_csv_keeps_message_when_timestamp_unparseable(tmp_path: Path):
    path = tmp_path / "reddit_bad_ts.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["author", "body", "created_utc"])
        writer.writeheader()
        writer.writerow({"author": "erin", "body": "a post with a broken timestamp", "created_utc": "not-a-number"})

    aliases = load_reddit_csv(str(path))
    assert len(aliases) == 1
    assert aliases[0].messages == ["a post with a broken timestamp"]
    assert aliases[0].post_hours_utc == []  # dropped silently, message kept
