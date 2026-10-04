#!/usr/bin/env python3
"""Move dialogues out of the pipeline so every later step (oracle, tokenize, package) skips them. Reversible.

Moves <data>/{audio,text,meta,oracle_raw,tokenized_audio,tokenized_text,tokenized_oracle}/<id>.* to
<data>_excluded/<same subdir>/. Nothing is deleted. Run it BEFORE oracle generation / tokenization; parquet
files that were already built are not touched (rebuild them with prepare_dataset + add_personaplex_prompt).
Stdlib only, run from the repo root.

Rules (combine freely; a dialogue is moved if ANY rule matches):
  --drop_emotional V1,V2   passenger voice in the list AND the passenger ever goes above intensity 0
                           (calm-only dialogues use only the neutral reference, so they are kept)
  --drop_voice V1,V2       every dialogue of these passenger voices
  --drop_ids FILE          ids, one per line (e.g. flagged by a later audit)
  --dest DIR               where to move them (default <data>_excluded); e.g. --dest data/cabin_eval for a hold-out
  --restore                move everything back (from --dest too)

    python3 -m tools.exclude_dialogues --data data/cabin --drop_emotional hero_02,hero_07 --dry_run
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

SUBDIRS = ("audio", "text", "meta", "oracle_raw", "tokenized_audio", "tokenized_text", "tokenized_oracle")


def peak_intensity(meta: dict) -> int:
    return max(((t.get("intensity") or 0) for t in meta.get("turns", []) if t.get("speaker") == "A"), default=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cabin")
    ap.add_argument("--drop_emotional", default="")
    ap.add_argument("--drop_voice", default="")
    ap.add_argument("--drop_ids", default=None)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    excl = Path(args.dest) if args.dest else data.parent / f"{data.name}_excluded"

    if args.restore:
        n = 0
        for sub in SUBDIRS:
            for f in sorted((excl / sub).glob("*")) if (excl / sub).exists() else []:
                if not args.dry_run:
                    (data / sub).mkdir(exist_ok=True)
                    shutil.move(str(f), str(data / sub / f.name))
                n += 1
        print(f"restored {n} files from {excl}" + ("  [DRY RUN]" if args.dry_run else ""))
        return

    emo = {v.strip() for v in args.drop_emotional.split(",") if v.strip()}
    anyv = {v.strip() for v in args.drop_voice.split(",") if v.strip()}
    ids = set(Path(args.drop_ids).read_text().split()) if args.drop_ids else set()

    drop, why = [], Counter()
    for mf in sorted((data / "meta").glob("*.json")):
        meta = json.loads(mf.read_text())
        vid = meta.get("passenger_voice_id")
        if mf.stem in ids:
            drop.append(mf.stem); why["id list"] += 1
        elif vid in anyv:
            drop.append(mf.stem); why[f"{vid} (all)"] += 1
        elif vid in emo and peak_intensity(meta) >= 1:
            drop.append(mf.stem); why[f"{vid} (emotional)"] += 1

    kept = len(list((data / "meta").glob("*.json"))) - len(drop)
    print(f"moving {len(drop)} dialogues, keeping {kept}   by rule: {dict(why)}"
          + ("   [DRY RUN]" if args.dry_run else ""))
    if args.dry_run:
        return
    moved = 0
    for did in drop:
        for sub in SUBDIRS:
            for f in (data / sub).glob(f"{did}.*") if (data / sub).exists() else []:
                (excl / sub).mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(excl / sub / f.name))
                moved += 1
    print(f"moved {moved} files to {excl}  (undo: --restore)")


if __name__ == "__main__":
    main()