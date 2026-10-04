#!/usr/bin/env python3
"""Find cross-talk dialogues: both speakers talking over each other for long stretches (stdlib only).

The oracle pilot showed dialogues whose words alternate A/B every 1-3 words (40-53 speaker switches per 100
words vs 5-9 in normal dialogues). Their oracle hints become fragments ("more", "down the") and the audio is
continuous overlap, so they should not be training data.

    python3 -m tools.check_overlap --data data/cabin                      # report + overlap_ids.txt
    python3 -m tools.check_overlap --data data/cabin --explain cc_000601  # turn timeline from meta/, why?
    python3 -m tools.exclude_dialogues --data data/cabin --drop_ids data/cabin/analysis/overlap_ids.txt

Measures per dialogue (from text/<id>.json word timings):
  switches_per_100w  speaker changes in the time-ordered word list, per 100 words
  overlap_frac       seconds where BOTH speakers are speaking / seconds where EITHER is speaking
Flagged when switches_per_100w > --max_switches (default 25) or overlap_frac > --max_overlap (default 0.20).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def merged(intervals: list[tuple[float, float]], gap: float = 0.3) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def total(iv: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in iv)


def overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    i = j = 0
    acc = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        acc += max(0.0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return acc


def measure(words: list[dict]) -> dict:
    words = sorted(words, key=lambda w: (w["start"], w["speaker"]))
    n = len(words)
    sw = sum(1 for a, b in zip(words, words[1:]) if a["speaker"] != b["speaker"])
    seg = {s: merged([(w["start"], w["end"]) for w in words if w["speaker"] == s]) for s in ("A", "B")}
    ov = overlap(seg["A"], seg["B"])
    union = total(seg["A"]) + total(seg["B"]) - ov
    return {"words": n, "switches_per_100w": 100 * sw / max(1, n), "overlap_s": ov,
            "overlap_frac": ov / union if union > 0 else 0.0}


def explain(data: Path, did: str) -> None:
    meta = json.loads((data / "meta" / f"{did}.json").read_text())
    print(f"{did}: duration {meta.get('duration_s')} s, {len(meta['turns'])} turns\n")
    print(f"{'#':>2} {'spk':3} {'start':>6} {'end':>6} {'dur':>5}  {'flags':16} {'overlap w/ prev':>15}  text")
    prev_end = None
    for i, t in enumerate(meta["turns"]):
        flags = ",".join(k for k in ("backchannel", "interrupted") if t.get(k)) or "-"
        ov = "" if prev_end is None else f"{max(0.0, prev_end - t['start']):.1f}s"
        mark = ""
        if prev_end is not None and prev_end - t["start"] > 0.5 and flags == "-":
            mark = "  <-- overlaps without a backchannel/interrupt flag"
        print(f"{i:2d} {t['speaker']:3} {t['start']:6.1f} {t['end']:6.1f} {t['end'] - t['start']:5.1f}  "
              f"{flags:16} {ov:>15}  {t['text'][:48]!r}{mark}   trigger={t.get('trigger')!r}")
        prev_end = max(prev_end or 0.0, t["end"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cabin")
    ap.add_argument("--max_switches", type=float, default=25.0)
    ap.add_argument("--max_overlap", type=float, default=0.20)
    ap.add_argument("--explain", default=None, metavar="DIALOGUE_ID")
    args = ap.parse_args()
    data = Path(args.data)
    if args.explain:
        explain(data, args.explain)
        return

    rows = []
    for tp in sorted((data / "text").glob("*.json")):
        m = measure(json.loads(tp.read_text()))
        meta_p = data / "meta" / tp.name
        nbc = 0
        if meta_p.exists():
            nbc = sum(1 for t in json.loads(meta_p.read_text()).get("turns", []) if t.get("backchannel"))
        rows.append({"id": tp.stem, "n_backchannel": nbc, **m,
                     "flag": m["switches_per_100w"] > args.max_switches or m["overlap_frac"] > args.max_overlap})
    out = data / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "overlap_report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    flagged = [r["id"] for r in rows if r["flag"]]
    (out / "overlap_ids.txt").write_text("\n".join(flagged) + ("\n" if flagged else ""))

    print(f"{len(rows)} dialogues, {len(flagged)} flagged (switches/100w > {args.max_switches} or overlap > {args.max_overlap:.0%})")
    bins = Counter(min(int(r["switches_per_100w"] // 5) * 5, 60) for r in rows)
    print("\nswitches per 100 words:")
    for b in sorted(bins):
        print(f"  {b:>2}{'+' if b == 60 else '-' + str(b + 4):4} {'#' * min(60, bins[b]):60s} {bins[b]}")
    print("\nflag rate by number of backchannel turns:")
    for k in sorted({r["n_backchannel"] for r in rows}):
        g = [r for r in rows if r["n_backchannel"] == k]
        print(f"  {k} backchannels: {sum(r['flag'] for r in g):3d}/{len(g):3d} flagged")
    print("\nworst 10:")
    for r in sorted(rows, key=lambda r: -r["switches_per_100w"])[:10]:
        print(f"  {r['id']}  switches/100w {r['switches_per_100w']:5.1f}  overlap {r['overlap_frac']:.0%}  backchannels {r['n_backchannel']}")
    print(f"\nwrote {out / 'overlap_report.csv'} and {out / 'overlap_ids.txt'}")


if __name__ == "__main__":
    main()
