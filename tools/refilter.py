#!/usr/bin/env python3
"""Per-voice, reversible voice-consistency filter (replaces `filter_voice_consistency --quarantine`).

Why a new one: filter_voice_consistency.py only ever ADDS quarantines, uses one global threshold set,
and lets dialogues through that have no similarity scores. This one re-judges every dialogue on every run:
  * a dialogue that now passes is RELEASED     (manifest.rejected.json -> manifest.json)
  * a dialogue that now fails is QUARANTINED   (manifest.json -> manifest.rejected.json)
  * thresholds can differ per voice (--override)
  * dialogues with missing / incomplete similarity scores are quarantined: a resumed render only
    scores the turns it re-rendered, so the old turns would silently escape the check
  * a dialogue that is already assembled (meta/<id>.json) is never quarantined
Same three rules as the original: min similarity, spread (max - min), emotional-vs-calm drop.
Writes turns/_voice_rejects.json in the original format (analyze_dataset.py reads it).

Stdlib only; run from the repo root. Thresholds are  min_sim,max_spread,max_emotion_drop :
    python3 -m tools.refilter --data data/cabin --default 0.75,0.15,0.08 --dry_run
    python3 -m tools.refilter --data data/cabin --default 0.75,0.15,0.08 --override hero_02=0.78,0.12,0.06

ALWAYS use this instead of `filter_voice_consistency --quarantine` from now on (run_overnight.sh calls the
old one, which would re-quarantine what you released).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


def metrics(man: dict) -> dict:
    """Passenger similarity statistics of one dialogue manifest (same split as the original filter)."""
    sims = man.get("speaker_similarity") or {}
    calm, emo, n_pax = [], [], 0
    for i, t in enumerate(man["script"]["turns"]):
        if t.get("speaker") != "passenger":
            continue
        n_pax += 1
        if str(i) not in sims:
            continue
        s = float(sims[str(i)])
        (calm if t.get("emotion") in (None, "neutral") or (t.get("intensity") or 0) == 0 else emo).append(s)
    allv = calm + emo
    if not allv:
        return {"scored": 0, "passenger_turns": n_pax, "min": None, "spread": None, "drop": None}
    return {"scored": len(allv), "passenger_turns": n_pax, "min": min(allv), "spread": max(allv) - min(allv),
            "drop": (fmean(calm) - fmean(emo)) if calm and emo else 0.0}


def violation(m: dict, thr: tuple[float, float, float]) -> str | None:
    if m["scored"] == 0:
        return "no similarity scores"
    if m["scored"] < m["passenger_turns"]:
        return f"incomplete scores {m['scored']}/{m['passenger_turns']}"
    min_sim, max_spread, max_drop = thr
    if m["min"] < min_sim:
        return f"min {m['min']:.2f}"
    if m["spread"] > max_spread:
        return f"spread {m['spread']:.2f}"
    if m["drop"] > max_drop:
        return f"emotional {m['drop']:.2f}"
    return None


def parse_thr(s: str) -> tuple[float, float, float]:
    a, b, c = (float(x) for x in s.split(","))
    return a, b, c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cabin")
    ap.add_argument("--default", default="0.78,0.12,0.06", help="min_sim,max_spread,max_emotion_drop "
                    "(default = the original strict filter)")
    ap.add_argument("--override", action="append", default=[], metavar="VOICE=min,spread,drop")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    turns, meta = data / "turns", data / "meta"
    default = parse_thr(args.default)
    over = {}
    for o in args.override:
        v, t = o.split("=")
        over[v] = parse_thr(t)

    before, after, total = Counter(), Counter(), Counter()
    rejected, changes, warns = [], Counter(), []
    n = 0
    for d in sorted(p for p in turns.iterdir() if p.is_dir()):
        ok_f, rej_f = d / "manifest.json", d / "manifest.rejected.json"
        if ok_f.exists() and rej_f.exists():
            warns.append(f"{d.name}: both manifest.json and manifest.rejected.json exist, skipped")
            continue
        cur = ok_f if ok_f.exists() else rej_f if rej_f.exists() else None
        if cur is None:
            continue  # failed render (no manifest): not a filter matter
        man = json.loads(cur.read_text())
        vid = man["script"]["passenger"]["voice_id"]
        why = violation(metrics(man), over.get(vid, default))
        accepted_now = cur == ok_f
        if why and accepted_now and (meta / f"{d.name}.json").exists():
            warns.append(f"{d.name}: fails ({why}) but is already assembled, left in place")
            why = None
        n += 1
        total[vid] += 1
        before[vid] += accepted_now
        after[vid] += why is None
        if why:
            rejected.append({"id": d.name, "voice": vid, "reason": why})
        if why and accepted_now:
            changes["quarantined"] += 1
            if not args.dry_run:
                ok_f.rename(rej_f)
        elif not why and not accepted_now:
            changes["released"] += 1
            if not args.dry_run:
                rej_f.rename(ok_f)

    print(f"{'voice':12s} {'dialogues':>9s} {'accepted before':>16s} {'accepted after':>15s}   thresholds")
    for v in sorted(total, key=lambda v: after[v] / total[v]):
        t = over.get(v, default)
        print(f"{v:12s} {total[v]:9d} {before[v]:16d} {after[v]:15d}   {t[0]:.2f},{t[1]:.2f},{t[2]:.2f}"
              + ("  (override)" if v in over else ""))
    print(f"\nchecked {n}: accepted {sum(after.values())} (was {sum(before.values())}), "
          f"released {changes['released']}, quarantined {changes['quarantined']}"
          + ("   [DRY RUN - nothing renamed]" if args.dry_run else ""))
    reasons = Counter(r["reason"].split()[0] for r in rejected)
    print("still rejected by rule:", dict(reasons))
    for w in warns[:15]:
        print("WARNING:", w)
    if len(warns) > 15:
        print(f"... {len(warns) - 15} more warnings")
    if not args.dry_run:
        by_voice = Counter(r["voice"] for r in rejected)
        (turns / "_voice_rejects.json").write_text(json.dumps(
            {"checked": n, "rejected": rejected, "by_voice": dict(by_voice.most_common())}, indent=1))


if __name__ == "__main__":
    main()
