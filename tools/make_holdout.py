#!/usr/bin/env python3
"""Pick a small stratified hold-out set for the oracle probes (stdlib only).

    python3 -m tools.make_holdout --data data/cabin --n 25
    python3 -m tools.exclude_dialogues --data data/cabin --drop_ids data/cabin/analysis/holdout_ids.txt --dest data/cabin_eval

Strata: scenario group (quota proportional to size, at least 1 each), then voices rotate inside a group so the
set covers many voices. Only dialogues that have audio, text, meta AND an oracle file are eligible. Deterministic
(--seed). Run it AFTER the full oracle run and BEFORE tokenizing, so hold-out dialogues never reach training.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cabin")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data, rng = Path(args.data), random.Random(args.seed)

    pool = defaultdict(list)
    for mp in sorted((data / "meta").glob("*.json")):
        did = mp.stem
        if not all((data / sub / f"{did}.{ext}").exists() for sub, ext in (("audio", "wav"), ("text", "json"), ("oracle_raw", "json"))):
            continue
        meta = json.loads(mp.read_text())
        peak = max(((t.get("intensity") or 0) for t in meta.get("turns", []) if t.get("speaker") == "A"), default=0)
        pool[(meta.get("scenario") or {}).get("group", "?")].append(
            (did, meta.get("passenger_voice_id"), (meta.get("scenario") or {}).get("category", ""), peak))
    total = sum(len(v) for v in pool.values())
    if not total:
        raise SystemExit("no eligible dialogues (need audio, text, meta and oracle_raw)")

    quota = {g: max(1, round(args.n * len(v) / total)) for g, v in pool.items()}
    while sum(quota.values()) > args.n:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < args.n:
        quota[max(pool, key=lambda g: len(pool[g]) - quota[g])] += 1

    picked = []
    for g, items in sorted(pool.items()):
        rng.shuffle(items)
        by_voice = defaultdict(list)
        for it in items:
            by_voice[it[1]].append(it)
        voices = sorted(by_voice, key=lambda v: rng.random())
        k = 0
        while len([p for p in picked if p[0] == g]) < quota[g] and any(by_voice.values()):
            v = voices[k % len(voices)]
            if by_voice[v]:
                picked.append((g, *by_voice[v].pop()))
            k += 1
    out = data / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_ids.txt").write_text("\n".join(p[1] for p in picked) + "\n")
    print(f"{len(picked)} hold-out dialogues of {total} eligible -> {out / 'holdout_ids.txt'}")
    print("groups:", dict(Counter(p[0] for p in picked)), "  voices:", len({p[2] for p in picked}),
          "  peak intensity:", sorted(Counter(p[4] for p in picked).items()))
    for p in sorted(picked)[:30]:
        print(f"  {p[1]}  {p[0]:9s} {p[2]:9s} {p[3]:28s} peak {p[4]}")


if __name__ == "__main__":
    main()
