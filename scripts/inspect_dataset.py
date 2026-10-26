"""Look inside a downloaded dataset (a .zip, or a folder) WITHOUT loading it all.

    python scripts/inspect_dataset.py C:\\caminho\\data-and-readme.zip

For every file it prints the name and size; for tables (.csv, .tsv, .json,
.jsonl) it prints the column names, the first rows (message text cut at 120
characters) and a guess at which columns are the author, the text and the
timestamp. Text files such as README are printed in full (first 3000 chars).

Why: before writing a loader we need to know the real column names, whether
each message has a timestamp, and how many users have enough text for the
alter-ego split. Nothing here is written to disk and nothing is uploaded.

PRIVACY: the data comes from real people on a dark web forum. Do not paste
rows of it into chats, issues or commits. Share only column names and counts.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import zipfile
from collections import Counter

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TABLE_EXT = (".csv", ".tsv", ".json", ".jsonl", ".ndjson")
TEXT_EXT = (".md", ".txt", ".rst", ".py")
MAX_ROWS_COUNTED = 2_000_000

AUTHOR_HINTS = ("author", "user", "poster", "member", "nick", "alias", "vendor", "sender")
TEXT_HINTS = ("content", "body", "text", "message", "post", "comment")
TIME_HINTS = ("time", "date", "created", "stamp", "posted")


def _guess(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    return [c for c in columns if any(h in c.lower() for h in hints)]


def _clip(value, n: int = 120) -> str:
    s = str(value).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[:n] + "..."


def _iter_rows(name: str, stream):
    lower = name.lower()
    if lower.endswith((".json",)):
        data = json.load(io.TextIOWrapper(stream, encoding="utf-8", errors="replace"))
        rows = data if isinstance(data, list) else data.get("data", [data]) if isinstance(data, dict) else []
        for r in rows:
            if isinstance(r, dict):
                yield r
    elif lower.endswith((".jsonl", ".ndjson")):
        for line in io.TextIOWrapper(stream, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    yield r
    else:
        delim = "\t" if lower.endswith(".tsv") else ","
        yield from csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", errors="replace", newline=""), delimiter=delim)


def describe_table(name: str, opener) -> None:
    try:
        with opener() as stream:
            rows = _iter_rows(name, stream)
            first = []
            columns: list[str] = []
            n = 0
            counters: dict[str, Counter] = {}
            author_cols: list[str] = []
            for r in rows:
                if n == 0:
                    columns = list(r.keys())
                    author_cols = _guess(columns, AUTHOR_HINTS)
                    counters = {c: Counter() for c in author_cols[:2]}
                if n < 3:
                    first.append(r)
                for c, ctr in counters.items():
                    ctr[str(r.get(c))] += 1
                n += 1
                if n >= MAX_ROWS_COUNTED:
                    break
    except Exception as exc:  # a malformed file must not stop the whole report
        print(f"    could not read as a table: {type(exc).__name__}: {exc}")
        return
    if not columns:
        print("    (empty)")
        return
    cap = " (stopped counting here)" if n >= MAX_ROWS_COUNTED else ""
    print(f"    rows: {n}{cap}")
    print(f"    columns ({len(columns)}): {', '.join(columns)}")
    print(f"    probable author column(s): {_guess(columns, AUTHOR_HINTS) or '?'}")
    print(f"    probable text column(s)  : {_guess(columns, TEXT_HINTS) or '?'}")
    print(f"    probable time column(s)  : {_guess(columns, TIME_HINTS) or '?'}")
    for i, r in enumerate(first):
        print(f"    row {i + 1}: " + " | ".join(f"{k}={_clip(v)}" for k, v in list(r.items())[:8]))
    for c, ctr in counters.items():
        if ctr:
            enough = sum(1 for v in ctr.values() if v >= 30)
            print(f"    '{c}': {len(ctr)} distinct values; {enough} appear on >= 30 rows "
                  f"(the paper needs >= 30 timestamped posts per user, plus 1500 words)")


def describe(name: str, size: int, opener) -> None:
    print(f"\n- {name}  ({size / 1e6:.1f} MB)")
    lower = name.lower()
    if lower.endswith(TABLE_EXT):
        describe_table(name, opener)
    elif lower.endswith(TEXT_EXT):
        with opener() as stream:
            print("    " + stream.read(3000).decode("utf-8", errors="replace").replace("\n", "\n    "))


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                p = os.path.join(root, f)
                describe(os.path.relpath(p, path), os.path.getsize(p), lambda p=p: open(p, "rb"))
    else:
        with zipfile.ZipFile(path) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            print(f"{path}: {len(infos)} files")
            for i in infos:
                describe(i.filename, i.file_size, lambda i=i: z.open(i))


if __name__ == "__main__":
    main()
