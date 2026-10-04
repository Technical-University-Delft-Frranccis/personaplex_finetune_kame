#!/usr/bin/env python3
"""Listening kit: hear a voice's references and samples of accepted / borderline / bad dialogues.

Run in the kame env (needs numpy + soundfile) from the repo root, after tools/refilter.py is in place:
    uv run --no-sync -m tools.make_listening_set --voices hero_02,hero_09,hero_07,hero_10

Writes to data/cabin/analysis/listen/ (per voice):
    <voice>_0_REFS.wav/.txt     neutral anchor + every emotion reference, a beep before each (timestamps in .txt)
    <voice>_1_OK_<id>           a dialogue that PASSED the strict filter (control; high intensity preferred)
    <voice>_2_A_<id>            rejected, but passes tier A thresholds   -> borderline
    <voice>_3_B_<id>            rejected, passes tier B but not A        -> mid
    <voice>_4_C_<id>            rejected even at tier B                  -> worst
Audio = the PASSENGER's turns only, in order, 0.5 s apart. The .txt lists each turn (index, reference used,
similarity, emotion_intensity, text); the lowest and highest similarity turns are marked.
Also INDEX.txt and verdicts.csv (a sheet to fill in while listening).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.refilter import metrics, parse_thr, violation

SR = 24_000


def read(path) -> np.ndarray:
    w, sr = sf.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        n = int(round(len(w) * SR / sr))
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype("float32")
    return w


def silence(s: float) -> np.ndarray:
    return np.zeros(int(s * SR), dtype="float32")


def beep() -> np.ndarray:
    t = np.arange(int(0.12 * SR)) / SR
    return (0.12 * np.sin(2 * np.pi * 880 * t)).astype("float32")


def ref_reel(vid: str, entry: dict, out: Path) -> bool:
    refs = entry["refs"]
    keys = ["neutral"] + sorted(k for k in refs if k != "neutral")
    parts, lines, t = [], [], 0.0
    for k in keys:
        p = Path(refs[k]["wav"])
        if not p.exists():
            lines.append(f"        {k:14s} MISSING FILE {p}")
            continue
        seg = np.concatenate([beep(), silence(0.3), read(p), silence(0.9)])
        lines.append(f"{t:6.1f}s  {k:14s} registry sim {refs[k].get('similarity', '?')}  "
                     f"source {refs[k].get('source', '?'):15s} text: {refs[k].get('text', '')[:55]}")
        parts.append(seg)
        t += len(seg) / SR
    if not parts:
        return False
    sf.write(out / f"{vid}_0_REFS.wav", np.concatenate(parts), SR)
    (out / f"{vid}_0_REFS.txt").write_text(f"{vid}: {entry.get('description', '')}\n\n" + "\n".join(lines))
    return True


def passenger_audio(d: Path, man: dict):
    sims = man.get("speaker_similarity") or {}
    parts, rows = [], []
    for i, t in enumerate(man["script"]["turns"]):
        if t.get("speaker") != "passenger" or not (d / f"{i:03d}.wav").exists():
            continue
        parts += [read(d / f"{i:03d}.wav"), silence(0.5)]
        rows.append((i, man.get("refs", {}).get(str(i), "?"), sims.get(str(i)),
                     f"{t.get('emotion')}_{t.get('intensity')}", t.get("text", "")))
    return (np.concatenate(parts) if parts else None), rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cabin")
    ap.add_argument("--voices", default="hero_02,hero_09,hero_07,hero_10")
    ap.add_argument("--per_tier", type=int, default=2)
    ap.add_argument("--tier_a", default="0.75,0.15,0.08")
    ap.add_argument("--tier_b", default="0.75,0.18,0.10")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    out = data / "analysis" / "listen"
    out.mkdir(parents=True, exist_ok=True)
    tier_a, tier_b = parse_thr(args.tier_a), parse_thr(args.tier_b)
    registry = json.loads((data / "voices.json").read_text())
    rng = random.Random(args.seed)

    pool = defaultdict(lambda: {"OK": [], "A": [], "B": [], "C": []})
    for d in sorted(p for p in (data / "turns").iterdir() if p.is_dir()):
        accepted = (d / "manifest.json").exists()
        mf = d / "manifest.json" if accepted else d / "manifest.rejected.json"
        if not mf.exists():
            continue
        man = json.loads(mf.read_text())
        vid, m = man["script"]["passenger"]["voice_id"], metrics(man)
        if accepted:
            peak = max((t.get("intensity") or 0) for t in man["script"]["turns"] if t["speaker"] == "passenger")
            if peak >= 2:
                pool[vid]["OK"].append((d, man, m))
        else:
            tier = "A" if violation(m, tier_a) is None else "B" if violation(m, tier_b) is None else "C"
            pool[vid][tier].append((d, man, m))

    print(f"{'voice':10s} {'OK':>5s} {'tierA':>6s} {'tierB':>6s} {'tierC':>6s}   "
          "(rejected dialogues recovered by tier A thresholds / by B-but-not-A / not even by B)")
    for v in sorted(pool):
        c = pool[v]
        if c["A"] or c["B"] or c["C"]:
            print(f"{v:10s} {len(c['OK']):5d} {len(c['A']):6d} {len(c['B']):6d} {len(c['C']):6d}")

    index, sheet = [], []
    for vid in [v.strip() for v in args.voices.split(",") if v.strip()]:
        if vid not in registry:
            print(f"skip {vid}: not in voices.json")
            continue
        ref_reel(vid, registry[vid], out)
        sheet.append([f"{vid}_0_REFS.wav", vid, "REFS", "", "", "", "", "", "", "", "", ""])
        for code, tag, k in (("1", "OK", 1), ("2", "A", args.per_tier), ("3", "B", args.per_tier),
                             ("4", "C", args.per_tier)):
            cands = pool[vid][tag]
            for d, man, m in rng.sample(cands, min(k, len(cands))):
                audio, rows = passenger_audio(d, man)
                if audio is None:
                    continue
                name = f"{vid}_{code}_{tag}_{d.name}"
                sf.write(out / f"{name}.wav", audio, SR)
                sims = [r[2] for r in rows if r[2] is not None]
                lines = []
                for i, ref, s, emo, text in rows:
                    mark = ""
                    if s is not None and sims and s == min(sims):
                        mark = "   <== LOWEST"
                    elif s is not None and sims and s == max(sims):
                        mark = "   <== HIGHEST"
                    ss = f"{s:.2f}" if s is not None else " n/a"
                    lines.append(f"{i:3d} {ref:13s} sim={ss} {emo:14s} {text}{mark}")
                (out / f"{name}.txt").write_text("\n".join(lines))
                sc = man["script"].get("scenario", {})
                index.append(f"{name}  min={m['min']:.2f} spread={m['spread']:.2f} drop={m['drop']:.2f}  "
                             f"{sc.get('category', '')} / {sc.get('arc', '')}")
                sheet.append([f"{name}.wav", vid, tag, f"{m['min']:.2f}", f"{m['spread']:.2f}",
                              f"{m['drop']:.2f}", sc.get("category", ""), "", "", "", "", ""])
    (out / "INDEX.txt").write_text("\n".join(index))
    with (out / "verdicts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "voice", "tier", "min", "spread", "drop", "category",
                    "same_person(S/D/X)", "emotion_ok(Y/N)", "artifacts(Y/N)", "worst_turn_idx", "notes"])
        w.writerows(sheet)
    print(f"\nwrote {len(sheet)} files + INDEX.txt + verdicts.csv to {out}")
    print(f"zip for download:  cd {data}/analysis && zip -r listen.zip listen")


if __name__ == "__main__":
    main()
