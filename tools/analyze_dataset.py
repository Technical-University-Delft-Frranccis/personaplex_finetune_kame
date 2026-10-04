#!/usr/bin/env python3
"""Analyse the cabin dialogue dataset: what survived, what was rejected, and why.

Stdlib only (no venv needed). Run from the repo root (personaplex_finetune_kame):

    python3 tools/analyze_dataset.py --data data/cabin --logs logs | tee analysis.txt

Status per script (first match wins):
    assembled          data/cabin/meta/<id>.json exists
    voice_rejected     turns/<id>/manifest.rejected.json   (filter_voice_consistency --quarantine)
    assembly_rejected  turns/<id>/manifest.json but no meta (assemble_dialogues quality_ok)
    tts_failed         turns/<id>/ exists but no manifest   (synthesize_turns REJECTED/ERROR/crash)
    not_rendered       no turns/<id>/ at all                (INVALID script, or shard died first)

Also writes <data>/analysis/per_dialogue.csv and <data>/analysis/rerender_candidates.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path

SCEN_KEYS = ("group", "category", "phase", "flight_type", "cabin_class", "outcome", "arc",
             "profanity", "opener", "modifier", "goal")
STATUSES = ("assembled", "voice_rejected", "assembly_rejected", "tts_failed", "not_rendered")
LINE = re.compile(r"^\[\d+/\d+\] (\S+): (.*?)(?: \(\d+s\))?$")


# ----------------------------------------------------------------------------- helpers
def load_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def wav_dur(p: Path) -> float | None:
    try:
        with wave.open(str(p)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def nwords(text: str) -> int:
    return len([w for w in (text or "").split() if any(c.isalpha() for c in w)])


def is_calm(t: dict) -> bool:  # same split as filter_voice_consistency.py
    return t["emotion"] in (None, "neutral") or t["intensity"] == 0


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def norm_reason(s: str) -> str:
    s = re.sub(r"\[[^\]]*\]", "[..]", s)
    return re.sub(r"\d+(\.\d+)?", "#", s)[:90]


def parse_status_logs(paths) -> tuple[dict, int]:
    status, oom = {}, 0
    for p in paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            if "out of memory" in line.lower():
                oom += 1
            m = LINE.match(line.strip())
            if m:
                status[m.group(1)] = m.group(2)  # last occurrence wins (latest run)
    return status, oom


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


# ----------------------------------------------------------------------------- loading
def load(args):
    data = Path(args.data)
    turns_dir, meta_dir = data / "turns", data / "meta"
    voices = load_json(data / "voices.json") or {}
    vrej = load_json(turns_dir / "_voice_rejects.json") or {}
    vrej_reason = {r["id"]: r["reason"] for r in vrej.get("rejected", [])}
    tts_status, tts_oom = parse_status_logs(sorted(Path(args.logs).glob("tts_*.log")))
    asm_status, _ = parse_status_logs([p for p in [Path(args.logs) / "assemble.log"] if p.exists()])

    recs = []
    for sp in sorted((data / "scripts").glob("cc_*.json")):
        s = load_json(sp)
        if s is None:
            recs.append({"id": sp.stem, "status": "not_rendered", "reason": "unreadable script json",
                         "turn_sims": [], "script": {}})
            continue
        did = s.get("id", sp.stem)
        ddir = turns_dir / did
        if (meta_dir / f"{did}.json").exists():
            status, reason = "assembled", ""
        elif (ddir / "manifest.rejected.json").exists():
            status, reason = "voice_rejected", vrej_reason.get(did, "?")
        elif (ddir / "manifest.json").exists():
            status, reason = "assembly_rejected", asm_status.get(did, "? (no line in assemble.log)")
        elif ddir.exists():
            status, reason = "tts_failed", tts_status.get(did, "? (no line in tts logs - crash/OOM?)")
        else:
            status, reason = "not_rendered", tts_status.get(did, "? (no line in tts logs)")

        scen, pas, crew = s.get("scenario") or {}, s.get("passenger") or {}, s.get("crew") or {}
        turns = s.get("turns") or []
        pturns = [t for t in turns if t.get("speaker") == "passenger"]
        vid = pas.get("voice_id")
        rec = {
            "id": did, "status": status, "reason": reason, "script": s,
            "voice": vid, "tier": (voices.get(vid) or {}).get("tier", "?"),
            "crew_voice": crew.get("voice_id"), "crew_performance": crew.get("performance"),
            "crew_mistake": crew.get("mistake"), "baseline_emotion": pas.get("baseline_emotion"),
            "peak_intensity": max(((t.get("intensity") or 0) for t in pturns), default=0),
            "n_turns": len(turns), "n_backchannel": sum(bool(t.get("backchannel")) for t in turns),
            "n_interrupt": sum(bool(t.get("interrupts")) for t in turns),
            "passenger_emotions": "|".join(sorted({str(t.get("emotion")) for t in pturns})),
            **{f"scen_{k}": scen.get(k) for k in SCEN_KEYS},
            "turn_sims": [], "duration_s": None,
        }
        manifest = load_json(ddir / "manifest.json") or load_json(ddir / "manifest.rejected.json")
        if manifest:
            sims, refs = manifest.get("speaker_similarity", {}), manifest.get("refs", {})
            for i, t in enumerate(manifest["script"]["turns"]):
                if t.get("speaker") != "passenger" or str(i) not in sims:
                    continue
                rec["turn_sims"].append({
                    "i": i, "sim": float(sims[str(i)]), "ref": refs.get(str(i), "?"),
                    "emotion": t.get("emotion"), "intensity": t.get("intensity") or 0,
                    "words": nwords(t.get("text", "")), "backchannel": bool(t.get("backchannel")),
                    "dur": wav_dur(ddir / f"{i:03d}.wav"), "text": t.get("text", ""),
                })
        if status == "assembled":
            rec["duration_s"] = (load_json(meta_dir / f"{did}.json") or {}).get("duration_s")
        recs.append(rec)
    return recs, voices, tts_oom


# ----------------------------------------------------------------------------- report sections
def funnel(recs, tts_oom):
    section("A. FUNNEL")
    c = Counter(r["status"] for r in recs)
    n = len(recs)
    for st in STATUSES:
        print(f"  {st:18s} {c[st]:5d}  {pct(c[st], n):>5s}")
    print(f"  {'total scripts':18s} {n:5d}")
    if tts_oom:
        print(f"  NOTE: {tts_oom} 'out of memory' lines in logs/tts_*.log")


def corpus(recs):
    section("B. ASSEMBLED CORPUS")
    a = [r for r in recs if r["status"] == "assembled"]
    durs = [r["duration_s"] for r in a if r["duration_s"]]
    if durs:
        print(f"  dialogues {len(a)}, total {sum(durs) / 3600:.2f} h, duration mean {mean(durs):.0f}s "
              f"min {min(durs):.0f}s max {max(durs):.0f}s")
        print(f"  < 20 s (dropped by add_personaplex_prompt): {sum(d < 20 for d in durs)}   "
              f"> 180 s (likely truncated at --max_frames 2400 incl. prompt): {sum(d > 180 for d in durs)}")
    nt = [r["n_turns"] for r in a]
    if nt:
        print(f"  turns/dialogue mean {mean(nt):.1f}; backchannels total {sum(r['n_backchannel'] for r in a)}, "
              f"interruptions total {sum(r['n_interrupt'] for r in a)}")

    # turn-level emotion x intensity, all scripts vs assembled
    def grid(rs):
        g = Counter()
        for r in rs:
            for t in (r["script"].get("turns") or []):
                if t.get("speaker") == "passenger":
                    g[(t.get("emotion"), t.get("intensity") or 0)] += 1
        return g

    ga, gall = grid(a), grid(recs)
    emos = sorted({e for e, _ in gall}, key=lambda e: -sum(v for (x, _), v in gall.items() if x == e))
    print("\n  passenger TURNS by emotion x intensity  (assembled / all scripts)")
    print(f"  {'emotion':12s}" + "".join(f"{'int ' + str(i):>14s}" for i in range(4)) + f"{'kept':>8s}")
    for e in emos:
        cells = "".join(f"{f'{ga[(e, i)]}/{gall[(e, i)]}':>14s}" for i in range(4))
        tot_a, tot = sum(ga[(e, i)] for i in range(4)), sum(gall[(e, i)] for i in range(4))
        print(f"  {str(e):12s}{cells}{pct(tot_a, tot):>8s}")


def breakdown(recs, key, top, label=None):
    by = defaultdict(Counter)
    for r in recs:
        by[str(r.get(key))][r["status"]] += 1
    rows = sorted(by.items(), key=lambda kv: -sum(kv[1].values()))
    print(f"\n  {label or key}  ({len(rows)} values)")
    print(f"  {'value':40s} {'total':>5s} {'asm':>5s} {'vrej':>5s} {'arej':>5s} {'ttsF':>5s} {'yield':>6s}")
    for val, c in rows[:top]:
        tot = sum(c.values())
        print(f"  {val[:40]:40s} {tot:5d} {c['assembled']:5d} {c['voice_rejected']:5d} "
              f"{c['assembly_rejected']:5d} {c['tts_failed'] + c['not_rendered']:5d} {pct(c['assembled'], tot):>6s}")
    if len(rows) > top:
        print(f"  ... {len(rows) - top} more values")


def distributions(recs, top):
    section("C. YIELD BY SCENARIO / CHARACTER FIELD  (asm=assembled vrej=voice-rejected arej=assembly-rejected ttsF=not rendered)")
    for k in SCEN_KEYS:
        breakdown(recs, f"scen_{k}", top, label=f"scenario.{k}")
    for k in ("crew_performance", "crew_mistake", "baseline_emotion", "peak_intensity", "tier",
              "passenger_emotions", "crew_voice"):
        breakdown(recs, k, top)


def voices_section(recs, voices):
    section("D. PASSENGER VOICES")
    by = defaultdict(list)
    for r in recs:
        by[r["voice"]].append(r)
    print(f"  {'voice':12s} {'tier':6s} {'n':>4s} {'asm':>4s} {'vrej':>4s} {'yield':>6s} "
          f"{'sim calm':>9s} {'sim emo':>8s} {'drop':>6s}  worst refs (mean sim, n)")
    rows = []
    for vid, rs in by.items():
        ts = [t for r in rs for t in r["turn_sims"]]
        calm = [t["sim"] for t in ts if is_calm(t)]
        emo = [t["sim"] for t in ts if not is_calm(t)]
        per_ref = defaultdict(list)
        for t in ts:
            per_ref[t["ref"]].append(t["sim"])
        worst = sorted(((mean(v), k, len(v)) for k, v in per_ref.items()))[:3]
        c = Counter(r["status"] for r in rs)
        rows.append((c["assembled"] / len(rs), vid, rs[0]["tier"], len(rs), c, calm, emo, worst))
    for y, vid, tier, n, c, calm, emo, worst in sorted(rows):
        w = ", ".join(f"{k} {m:.2f} ({k_n})" for m, k, k_n in worst)
        print(f"  {str(vid):12s} {tier:6s} {n:4d} {c['assembled']:4d} {c['voice_rejected']:4d} {y * 100:5.0f}% "
              f"{mean(calm):9.3f} {mean(emo):8.3f} {mean(calm) - mean(emo):6.3f}  {w}")

    print("\n  voice-bank emotion references for the 8 lowest-yield HERO voices "
          "(registry similarity to anchor, source) vs. mean similarity of turns rendered from that ref")
    heroes = [row for row in sorted(rows) if row[2] == "hero"][:8]
    for _, vid, *_ in heroes:
        ts = [t for r in by[vid] for t in r["turn_sims"]]
        per_ref = defaultdict(list)
        for t in ts:
            per_ref[t["ref"]].append(t["sim"])
        print(f"  {vid}:")
        for key, ref in sorted(((voices.get(vid) or {}).get("refs") or {}).items()):
            got = per_ref.get(key, [])
            print(f"     {key:14s} ref sim {ref.get('similarity', float('nan')):.3f} {ref.get('source', '?'):15s} "
                  f"turns: n={len(got):3d} mean {mean(got):.3f} min {min(got) if got else float('nan'):.3f}")

    print("\n  per reference key across ALL voices (is one emotion level bad everywhere?)")
    per_key = defaultdict(list)
    for r in recs:
        for t in r["turn_sims"]:
            per_key[t["ref"]].append(t["sim"])
    for key, v in sorted(per_key.items(), key=lambda kv: mean(kv[1])):
        lo = sorted(v)[max(0, len(v) // 10 - 1)]
        print(f"     {key:14s} n={len(v):5d} mean {mean(v):.3f}  p10 {lo:.3f}")


def rejection_section(recs):
    section("E. VOICE-CONSISTENCY REJECTIONS")
    vr = [r for r in recs if r["status"] == "voice_rejected"]
    kind = Counter(r["reason"].split()[0] if r["reason"] else "?" for r in vr)
    print(f"  {len(vr)} rejected; by rule: {dict(kind)}  (min=min similarity, spread=max-min, emotional=calm-vs-emo drop)")
    vals = defaultdict(list)
    for r in vr:
        m = re.search(r"(\d+\.\d+)", r["reason"])
        if m:
            vals[r["reason"].split()[0]].append(float(m.group(1)))
    for k, v in vals.items():
        v = sorted(v)
        print(f"     {k:10s} value median {v[len(v) // 2]:.3f}  range {v[0]:.3f}-{v[-1]:.3f}")

    print("\n  the LOWEST-similarity passenger turn of each rejected dialogue:")
    worst = [min(r["turn_sims"], key=lambda t: t["sim"]) for r in vr if r["turn_sims"]]
    n = len(worst)
    if n:
        print(f"     is a backchannel: {pct(sum(t['backchannel'] for t in worst), n)}   "
              f"<= 2 words: {pct(sum(t['words'] <= 2 for t in worst), n)}   "
              f"<= 4 words: {pct(sum(t['words'] <= 4 for t in worst), n)}   "
              f"clip < 1.0 s: {pct(sum((t['dur'] or 9) < 1.0 for t in worst), n)}")
        print(f"     intensity: {sorted(Counter(t['intensity'] for t in worst).items())}")
        print(f"     ref key:   {Counter(t['ref'] for t in worst).most_common(8)}")
        print("     examples (sim, ref, words, dur, text):")
        for t in sorted(worst, key=lambda t: t["sim"])[:12]:
            d = f"{t['dur']:.1f}s" if t["dur"] else "?"
            print(f"       {t['sim']:.3f} {t['ref']:12s} {t['words']:2d}w {d:>5s}  {t['text'][:70]!r}")

    section("F. SIMILARITY vs CLIP LENGTH / INTENSITY (all rendered passenger turns)")
    ts = [t for r in recs for t in r["turn_sims"]]

    def show(name, keyf, order):
        g = defaultdict(list)
        for t in ts:
            g[keyf(t)].append(t["sim"])
        print(f"  by {name}:")
        for k in order:
            if g[k]:
                v = sorted(g[k])
                print(f"     {k:12s} n={len(v):5d} mean {mean(v):.3f}  p10 {v[len(v) // 10]:.3f}  "
                      f"< 0.78: {pct(sum(x < 0.78 for x in v), len(v))}")

    wb = lambda t: "bc" if t["backchannel"] else ("1-2w" if t["words"] <= 2 else "3-5w" if t["words"] <= 5
                                                  else "6-10w" if t["words"] <= 10 else "11+w")
    show("length", wb, ["bc", "1-2w", "3-5w", "6-10w", "11+w"])
    show("intensity", lambda t: f"int {t['intensity']}", [f"int {i}" for i in range(4)])


def what_if(recs):
    section("G. WHAT-IF: filter thresholds (counts over all dialogues that have similarity data)")

    def passes(ts, min_sim, spread, drop, skip_short):
        use = [t for t in ts if not (skip_short and (t["backchannel"] or t["words"] <= skip_short))]
        calm = [t["sim"] for t in use if is_calm(t)]
        emo = [t["sim"] for t in use if not is_calm(t)]
        allv = calm + emo
        if not allv:
            return True
        if min(allv) < min_sim or max(allv) - min(allv) > spread:
            return False
        return not (calm and emo and mean(calm) - mean(emo) > drop)

    rs = [r for r in recs if r["turn_sims"] and r["status"] in ("assembled", "voice_rejected", "assembly_rejected")]
    vr = [r for r in rs if r["status"] == "voice_rejected"]
    grid = [(0.78, 0.12, 0.06, 0), (0.78, 0.12, 0.06, 2), (0.78, 0.12, 0.06, 4),
            (0.75, 0.15, 0.08, 0), (0.75, 0.15, 0.08, 2), (0.72, 0.18, 0.10, 0), (0.72, 0.18, 0.10, 2)]
    print(f"  {'min_sim':>7s} {'spread':>6s} {'drop':>5s} {'ignore<=w':>9s} {'pass(all)':>10s} {'recovered':>10s}  recovered by voice")
    for g in grid:
        ok = [r for r in rs if passes(r["turn_sims"], *g)]
        rec = [r for r in vr if passes(r["turn_sims"], *g)]
        top = Counter(r["voice"] for r in rec).most_common(4)
        print(f"  {g[0]:7.2f} {g[1]:6.2f} {g[2]:5.2f} {('bc+' + str(g[3])) if g[3] else 'none':>9s} "
              f"{len(ok):>5d}/{len(rs):<4d} {len(rec):>10d}  {top}")
    print("  (first row should reproduce the current filter; 'ignore<=w' drops backchannels and turns of <= w words")
    print("   from the check, since WavLM x-vectors on sub-second clips are noisy)")


def other_failures(recs):
    section("H. ASSEMBLY REJECTS AND TTS FAILURES")
    for st in ("assembly_rejected", "tts_failed", "not_rendered"):
        rs = [r for r in recs if r["status"] == st]
        print(f"  {st}: {len(rs)}")
        for reason, n in Counter(norm_reason(r["reason"]) for r in rs).most_common(10):
            print(f"     {n:4d}  {reason}")
        if st == "assembly_rejected":
            for r in rs[:10]:
                m = re.search(r"turn (\d+)", r["reason"])
                if m:
                    t = (r["script"].get("turns") or [{}] * 99)[int(m.group(1))]
                    print(f"       {r['id']} turn {m.group(1)} ({t.get('speaker')}, bc={bool(t.get('backchannel'))}): "
                          f"{str(t.get('text'))[:70]!r}")


def write_outputs(recs, data: Path):
    out = data / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "status", "reason", "voice", "tier", "crew_voice", "crew_performance", "crew_mistake",
            "baseline_emotion", "peak_intensity", "n_turns", "n_backchannel", "n_interrupt", "duration_s",
            "passenger_emotions", "min_sim", "mean_sim"] + [f"scen_{k}" for k in SCEN_KEYS]
    with (out / "per_dialogue.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            sims = [t["sim"] for t in r["turn_sims"]]
            w.writerow({**r, "min_sim": min(sims) if sims else "", "mean_sim": round(mean(sims), 3) if sims else ""})
    cand = {"voice_rejected": defaultdict(list), "assembly_rejected": [], "tts_failed": [], "not_rendered": []}
    for r in recs:
        if r["status"] == "voice_rejected":
            cand["voice_rejected"][r["voice"]].append({"id": r["id"], "reason": r["reason"]})
        elif r["status"] in cand:
            cand[r["status"]].append({"id": r["id"], "reason": r["reason"]})
    (out / "rerender_candidates.json").write_text(json.dumps(cand, indent=1))
    print(f"\nwrote {out / 'per_dialogue.csv'} and {out / 'rerender_candidates.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cabin")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--top", type=int, default=12, help="max rows per breakdown table")
    args = ap.parse_args()

    recs, voices, tts_oom = load(args)
    if not recs:
        raise SystemExit(f"no scripts found in {args.data}/scripts")
    funnel(recs, tts_oom)
    corpus(recs)
    distributions(recs, args.top)
    voices_section(recs, voices)
    rejection_section(recs)
    what_if(recs)
    other_failures(recs)
    write_outputs(recs, Path(args.data))


if __name__ == "__main__":
    main()
