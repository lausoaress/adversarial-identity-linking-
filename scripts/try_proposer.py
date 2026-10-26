"""Quick check of the Proposer with a REAL model, on 6 hand-written pairs.

    python scripts/try_proposer.py --provider ollama --model llama3.1 --timeout 600

The six pairs are invented test text (not real people). Three are the
same person, three are different people, with the clues a good Proposer
should pick up: matching stated facts (city, job, pet), a contradicted age,
a contradicted country, and one pair with nothing to go on.

What this tells you (it is a smoke test, six samples is NOT a result):
  - does the model answer in the JSON format, or do replies come back as
    `parse_error`?
  - does it quote the text, or do many evidence items get dropped by the
    quote check (`dropped_evidence`)? That is a direct count of how often
    the model invents evidence.
  - does the verdict go the way the clues point?
  - how long does one call take on this machine?

The baseline score is not shown to the model here (similarity_score=None),
so this also is the "single LLM agent, no cosine hint" setting.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.llm_client import build_client
from src.agents.proposer import ProposerAgent
from src.schemas import Alias, CandidatePair, Platform, ProposerVerdict

# (name, expected, dark messages, dark hours, open messages, open hours)
CASES = [
    (
        "A. same person - matching city, hobby and vendor complaint",
        "same",
        [
            "Ordered from that vendor again and tbh the quality was way below what the listing promised.",
            "Anyone in Miami want to split shipping? I am sick of waiting two weeks.",
            "Slept badly, going to yoga class in the morning to reset.",
            "That batch was weak. Same thing happened last month, I lost money on it.",
            "I work nights so I am usually online around 2 or 3 am.",
        ],
        [1, 2, 3, 2, 1],
        [
            "tbh the vendor I mentioned sent me a really weak batch, nothing like the listing.",
            "Living in Miami is great but shipping times here are awful.",
            "Yoga class this morning was exactly what I needed after another night shift.",
            "Posting from my phone at 3 am again, my job keeps me up all night.",
            "Lost a good chunk of money on that order, never buying from them again.",
        ],
        [2, 3, 3, 3, 2],
    ),
    (
        "B. same person - dog, nursing job, same closing habit",
        "same",
        [
            "Long shift at the hospital again, my feet hurt. Bruno ate my shoe, as usual.",
            "As a nurse I can tell you that mixing those is a bad idea. Stay safe.",
            "Took Bruno to the park, he is a golden retriever and will not stop running.",
            "Night shifts wreck my sleep. Anyway, cheers and take care.",
        ],
        [22, 23, 1, 4],
        [
            "Nurse here. Please do not mix those, I see the results in the ER every week.",
            "My golden retriever Bruno chewed up another pair of shoes today.",
            "Worked a double night shift, I can barely type. Anyway, cheers and take care.",
            "Walked Bruno at the park near the hospital before going to bed.",
        ],
        [0, 22, 5, 7],
    ),
    (
        "C. same person - style only, no stated facts",
        "same",
        [
            "ngl the new update is kinda mid, but whatever, it works i guess lol",
            "idk man, prices went up again?? kinda wild honestly",
            "lol yeah thats what i figured, ngl",
            "whatever works i guess, not gonna argue about it",
        ],
        [20, 21, 21, 22],
        [
            "ngl this release is kinda mid, but it works i guess lol",
            "idk, everything got more expensive again?? kinda wild honestly",
            "yeah lol thats what i figured",
            "not gonna argue about it, whatever works",
        ],
        [21, 22, 22, 23],
    ),
    (
        "D. different people - similar style, contradicted age",
        "different",
        [
            "I am 20 and still living with my parents, college is draining me.",
            "ngl the exam week is killing me, tbh I barely sleep.",
            "Anyone else my age tired of everything being so expensive?",
            "tbh I just want to graduate already lol",
        ],
        [19, 20, 21, 20],
        [
            "I am 34, been working in logistics for ten years now.",
            "ngl the end of quarter is killing me, tbh I barely sleep.",
            "Anyone else my age tired of everything being so expensive?",
            "tbh I just want a long vacation already lol",
        ],
        [19, 20, 20, 21],
    ),
    (
        "E. different people - contradicted country and habit",
        "different",
        [
            "Here in Poland the winter is brutal, I only go out for work.",
            "I never touch any drugs, I only come here to read the discussions.",
            "Just reading the forum threads again, good info as always.",
            "Winter in Warsaw again, so dark at 4 pm.",
        ],
        [17, 18, 19, 16],
        [
            "Fourth of July in Texas, everyone is grilling outside.",
            "I take that stuff every weekend, nothing beats it.",
            "Just reading the forum threads again, good info as always.",
            "Got my American passport renewed today, took forever.",
        ],
        [1, 2, 3, 0],
    ),
    (
        "F. different people - nothing in common",
        "different",
        [
            "Does anyone know a good guide for setting up PGP on Linux? Keeps failing.",
            "Thanks, that fixed it. The key export was the problem.",
            "I mostly lurk, I only post when I get stuck on tech stuff.",
        ],
        [10, 11, 15],
        [
            "Best pasta recipe? I want something quick for tonight's dinner.",
            "My grandmother used to make this with fresh basil, it was perfect.",
            "Anyone got tips for keeping tomatoes fresh on the balcony?",
        ],
        [18, 19, 8],
    ),
]


def build_pairs() -> list[tuple[str, str, CandidatePair]]:
    out = []
    for i, (name, expected, dm, dh, om, oh) in enumerate(CASES):
        dark = Alias(f"dark_{i}", Platform.DARK_WEB, "handwritten", dm, dh)
        open_ = Alias(f"open_{i}", Platform.OPEN_WEB, "handwritten", om, oh)
        out.append((name, expected, CandidatePair(dark, open_, similarity_score=None)))
    return out


def run(llm, verbose: bool = True) -> list[dict]:
    agent = ProposerAgent(llm=llm)
    rows = []
    for name, expected, pair in build_pairs():
        t0 = time.time()
        result = agent.run(pair)
        seconds = time.time() - t0
        rows.append({"name": name, "expected": expected, "result": result, "seconds": seconds})
        if verbose:
            kept = [e for e in result.evidence if e.source_agent == "proposer"]
            print(f"\n{name}")
            print(f"  expected: {expected}   verdict: {result.verdict.value}   "
                  f"confidence: {result.confidence:.2f}   ({seconds:.0f}s)")
            if result.parse_error:
                print("  PARSE ERROR:", "; ".join(result.warnings))
                print("  raw reply:", result.raw_response[:200].replace("\n", " "))
            else:
                print(f"  hypothesis: {result.hypothesis}")
                print(f"  evidence kept: {len(kept)}   dropped (quote not found in text): "
                      f"{len(result.dropped_evidence)}")
                for e in kept[:3]:
                    print(f"    - [{e.kind}] {e.description}")
                for d in result.dropped_evidence[:2]:
                    print(f"    x dropped: {d['description']}  (quote: {d['quote_dark'] or d['quote_open']!r})")
                for w in result.warnings:
                    print("  warning:", w)
    return rows


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    errors = sum(r["result"].parse_error for r in rows)
    kept = sum(len([e for e in r["result"].evidence if e.source_agent == "proposer"]) for r in rows)
    dropped = sum(len(r["result"].dropped_evidence) for r in rows)
    ok = 0
    for r in rows:
        v = r["result"].verdict
        if r["expected"] == "same" and v == ProposerVerdict.LINK:
            ok += 1
        elif r["expected"] == "different" and v == ProposerVerdict.NO_LINK:
            ok += 1
    unsure = sum(r["result"].verdict == ProposerVerdict.UNSURE and not r["result"].parse_error for r in rows)
    total_ev = kept + dropped
    print("\n=== Summary (6 hand-written pairs: a smoke test, not a result) ===")
    print(f"Parse errors            : {errors}/{n}")
    print(f"Verdict as expected     : {ok}/{n}   (unsure: {unsure})")
    if total_ev:
        print(f"Evidence dropped        : {dropped}/{total_ev} ({dropped / total_ev:.0%}) - quote not found in the text")
    print(f"Average time per call   : {sum(r['seconds'] for r in rows) / n:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai", "anthropic"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=None, help="seconds per Ollama call (default 300)")
    args = parser.parse_args()

    kwargs = {"model": args.model} if args.model else {}
    if args.provider == "ollama" and args.timeout is not None:
        kwargs["timeout_s"] = args.timeout
    llm = build_client(args.provider, **kwargs)
    print(f"Provider: {args.provider}  model: {args.model or 'default'}")
    summarize(run(llm))


if __name__ == "__main__":
    main()
