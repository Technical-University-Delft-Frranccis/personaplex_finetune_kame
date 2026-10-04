#!/usr/bin/env python3
"""Stage a re-render batch (stdlib only, run from the repo root).

Selects (never anything already assembled, i.e. with meta/<id>.json, and never an accepted-but-unassembled one):
  quarantined   turns/<id>/manifest.rejected.json        voice-consistency rejects
  failed        turns/<id>/ with no manifest at all      synthesize_turns rejected a turn / crashed
  missing       scripts/<id>.json with no turns/<id>/    never rendered (may be an INVALID script)

Then:
  * --reassign SRC=T1,T2   the passenger voice of every selected SRC dialogue becomes T1,T2,T1,... (round robin).
                           The brief was written for SRC's profile, so pick targets of the SAME gender and about the
                           same age/accent. Originals are saved in scripts_pre_reassign/.
  * --skip VOICE           do not touch this voice's dialogues at all (leave them rejected)
  * deletes turns/<id> for every selected dialogue. Required: a half-finished dir would be "resumed" and its
    manifest would only score the newly rendered turns; a dir with all wavs would be "completed" with NO scores.
  * copies the selected scripts to scripts_rerender/ so synthesize_turns can be pointed at exactly this batch
    (it never touches quarantined dialogues that were left out).

    python3 -m tools.prepare_rerender --data data/cabin --reassign hero_02=hero_06,hero_10 --dry_run
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cabin")
    ap.add_argument("--reassign", action="append", default=[], metavar="SRC=T1,T2")
    ap.add_argument("--skip", action="append", default=[], metavar="VOICE")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    scripts_dir, turns, meta = data / "scripts", data / "turns", data / "meta"
    voices = json.loads((data / "voices.json").read_text())
    desc = lambda v: f"{v} [{voices[v].get('tier')}] {voices[v].get('description', '')}"

    reassign = {}
    for item in args.reassign:
        src, tgt = item.split("=")
        targets = [t.strip() for t in tgt.split(",") if t.strip()]
        for t in targets:
            if t not in voices or voices[t].get("role") != "passenger" or voices[t].get("tier") != "hero":
                raise SystemExit(f"--reassign target {t!r} must be a hero passenger voice in voices.json "
                                 "(pool voices have no emotion references)")
        reassign[src] = targets

    selected = []  # (path, script, kind)
    for sp in sorted(scripts_dir.glob("cc_*.json")):
        s = json.loads(sp.read_text())
        did, vid = s.get("id", sp.stem), (s.get("passenger") or {}).get("voice_id")
        if (meta / f"{did}.json").exists() or vid in args.skip:
            continue
        d = turns / did
        if (d / "manifest.json").exists():
            continue
        kind = "quarantined" if (d / "manifest.rejected.json").exists() else "failed" if d.exists() else "missing"
        selected.append((sp, s, kind))

    counter, changed = Counter(), []
    for sp, s, kind in selected:
        vid = s["passenger"]["voice_id"]
        if vid in reassign:
            new = reassign[vid][counter[vid] % len(reassign[vid])]
            counter[vid] += 1
            changed.append((sp, s, vid, new))

    print("selected:", dict(Counter(k for _, _, k in selected)), f"total {len(selected)}")
    print("by voice:", Counter(s["passenger"]["voice_id"] for _, s, _ in selected).most_common())
    for src, targets in reassign.items():
        n = sum(1 for _, s, k in selected if s["passenger"]["voice_id"] == src)
        print(f"\nreassign {n} dialogues:\n   from {desc(src)}")
        for t in targets:
            print(f"   to   {desc(t)}")
    if args.dry_run:
        print("\n[DRY RUN - nothing changed]")
        return

    if changed:
        backup = data / "scripts_pre_reassign"
        backup.mkdir(exist_ok=True)
        for sp, s, old, new in changed:
            if not (backup / sp.name).exists():
                shutil.copy2(sp, backup / sp.name)
            s["passenger"]["voice_id"] = new
            sp.write_text(json.dumps(s, indent=1, ensure_ascii=False))
    stage = data / "scripts_rerender"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    removed = 0
    for sp, s, kind in selected:
        did = s.get("id", sp.stem)
        if (turns / did).exists():
            shutil.rmtree(turns / did)
            removed += 1
        shutil.copy2(sp, stage / sp.name)
    print(f"\nremoved {removed} turns dirs, staged {len(selected)} scripts in {stage}, "
          f"reassigned {len(changed)} (originals in scripts_pre_reassign/)")


if __name__ == "__main__":
    main()
