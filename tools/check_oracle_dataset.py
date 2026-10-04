#!/usr/bin/env python3
"""Quality gate for the FULL oracle run (stdlib only, needs tools/oracle_text.py next to it).

    python3 -m tools.check_oracle_dataset --data data/cabin

Reads <data>/oracle_raw/*.json and reports, over LLM-written predictions only (verbatim hints excluded):
  ALL-CAPS share, lowercase-start share, crew-voice share (tools.oracle_text.CREW_CUES), 1-2 word predictions;
plus schema / hint-gating violations, fragment hints (<= 2 words: a sign of cross-talk dialogues), empty files,
and which dialogues are outliers. Writes <data>/analysis/oracle_outliers.txt (ids you may want to exclude).

Healthy run (pilot targets): all-caps ~0%, lowercase start < 5%, crew-voice < 5%, no schema violations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.oracle_text import looks_like_crew

KEYS = ["timestamp_ms", "conversation_context", "prediction", "total_word_count", "trigger_word",
        "recent_words", "current_spoken_ratio", "channel", "hint"]


def is_caps(p: str) -> bool:
    letters = [c for c in p if c.isalpha()]
    return len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cabin")
    ap.add_argument("--oracle_dir", default=None, help="default <data>/oracle_raw")
    ap.add_argument("--max_caps", type=float, default=0.05)
    ap.add_argument("--max_crew", type=float, default=0.15)
    ap.add_argument("--max_fragment_hints", type=float, default=0.5)
    args = ap.parse_args()
    data = Path(args.data)
    odir = Path(args.oracle_dir) if args.oracle_dir else data / "oracle_raw"
    files = sorted(odir.glob("*.json"))
    text_ids = {p.stem for p in (data / "text").glob("*.json")}
    done_ids = {p.stem for p in files}

    tot = dict(events=0, llm=0, caps=0, low=0, crew=0, short=0, bad_schema=0, bad_hint=0, hint_events=0, verbatim=0)
    outliers: dict[str, list[str]] = {}
    empty = []
    for f in files:
        try:
            ev = json.loads(f.read_text())
        except Exception:
            outliers[f.stem] = ["unreadable json"]
            continue
        if not ev:
            empty.append(f.stem)
            continue
        llm = [e for e in ev if not e["hint"] or e["prediction"].strip() != e["hint"].strip()]
        c = dict(caps=sum(is_caps(e["prediction"]) for e in llm),
                 crew=sum(looks_like_crew(e["prediction"]) for e in llm))
        why = []
        if llm and c["caps"] / len(llm) > args.max_caps:
            why.append(f"all-caps {c['caps']}/{len(llm)}")
        if llm and c["crew"] / len(llm) > args.max_crew:
            why.append(f"crew-voice {c['crew']}/{len(llm)}")
        hints = sorted({e["hint"] for e in ev if e["hint"]})
        frag = [h for h in hints if len(h.split()) <= 2]
        if hints and len(frag) / len(hints) > args.max_fragment_hints and len(hints) >= 4:
            why.append(f"fragment hints {len(frag)}/{len(hints)} (cross-talk?)")
        if why:
            outliers[f.stem] = why
        tot["events"] += len(ev)
        tot["llm"] += len(llm)
        tot["caps"] += c["caps"]
        tot["crew"] += c["crew"]
        tot["low"] += sum(e["prediction"][:1].islower() for e in llm)
        tot["short"] += sum(len(e["prediction"].split()) <= 2 for e in llm)
        tot["bad_schema"] += sum(list(e.keys()) != KEYS for e in ev)
        tot["bad_hint"] += sum((e["current_spoken_ratio"] > 0.5) != bool(e["hint"]) for e in ev)
        tot["hint_events"] += sum(bool(e["hint"]) for e in ev)
        tot["verbatim"] += len(ev) - len(llm)

    n = max(1, tot["llm"])
    print(f"oracle files: {len(files)}   text transcripts: {len(text_ids)}   missing oracle: {len(text_ids - done_ids)}"
          f"   empty: {len(empty)}")
    print(f"events: {tot['events']}  (LLM-written {tot['llm']}, verbatim hint {tot['verbatim']} = "
          f"{tot['verbatim'] / max(1, tot['hint_events']):.0%} of hinted events)")
    print(f"ALL-CAPS        {tot['caps'] / n:6.1%}   (target ~0%)")
    print(f"lowercase start {tot['low'] / n:6.1%}   (target < 5%)")
    print(f"crew-voice cues {tot['crew'] / n:6.1%}   (target < 5%; the retry already removes what the cues see)")
    print(f"<= 2 words      {tot['short'] / n:6.1%}")
    print(f"schema violations {tot['bad_schema']}   hint-gating violations {tot['bad_hint']}   (both must be 0)")
    ids = sorted(outliers)
    out = data / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "oracle_outliers.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    print(f"\n{len(ids)} outlier dialogues (written to {out / 'oracle_outliers.txt'}):")
    for i in ids[:15]:
        print(f"  {i}: {'; '.join(outliers[i])}")
    if len(ids) > 15:
        print(f"  ... {len(ids) - 15} more")
    if text_ids - done_ids:
        print(f"\nWARNING: {len(text_ids - done_ids)} dialogues have no oracle file yet (still running, or errors in the log)")


if __name__ == "__main__":
    main()
