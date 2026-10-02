"""Step 1b - reject dialogues whose passenger voice drifts between turns (runs anywhere, no GPU).

synthesize_turns.py already rejects a TURN whose similarity to the voice anchor is below
--min_similarity. That still lets a dialogue through where calm turns score 0.92 and angry turns
0.76: every turn passes, but the voice audibly changes when the passenger gets angry. Train on
that and the model learns "escalation = different voice", which is exactly the accent/voice switch
we want to avoid at inference. This step looks at the whole dialogue instead:

  - min passenger similarity  >= --min_sim       (default 0.78, a bit above the per-turn gate)
  - spread (max - min)        <= --max_spread    (default 0.12)
  - the emotional turns' mean is within --max_emotion_drop of the calm turns' mean

Run between synthesize_turns.py and assemble_dialogues.py. With --quarantine the manifest of a
rejected dialogue is renamed, so assemble_dialogues.py skips it; re-render it with another seed or
fix the voice's emotion reference (a voice that fails often has a bad reference: listen to it).

    python -m tools.filter_voice_consistency --turns_dir data/cabin/turns --quarantine
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def check(manifest: dict, args) -> str | None:
    script, sims = manifest["script"], manifest.get("speaker_similarity", {})
    calm, emo = [], []
    for i, t in enumerate(script["turns"]):
        if t["speaker"] != "passenger" or str(i) not in sims:
            continue
        s = float(sims[str(i)])
        (calm if t.get("emotion") in (None, "neutral") or t.get("intensity", 0) == 0 else emo).append(s)
    allv = calm + emo
    if not allv:
        return None
    if min(allv) < args.min_sim:
        return f"min similarity {min(allv):.2f}"
    if max(allv) - min(allv) > args.max_spread:
        return f"spread {max(allv) - min(allv):.2f}"
    if calm and emo and sum(calm) / len(calm) - sum(emo) / len(emo) > args.max_emotion_drop:
        return f"emotional turns {sum(calm) / len(calm) - sum(emo) / len(emo):.2f} below calm turns"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--turns_dir", required=True)
    p.add_argument("--min_sim", type=float, default=0.78)
    p.add_argument("--max_spread", type=float, default=0.12)
    p.add_argument("--max_emotion_drop", type=float, default=0.06)
    p.add_argument("--quarantine", action="store_true", help="rename manifest.json of rejected dialogues")
    args = p.parse_args()

    bad_voices, n, rejected = Counter(), 0, []
    for m in sorted(Path(args.turns_dir).glob("*/manifest.json")):
        manifest = json.loads(m.read_text())
        n += 1
        why = check(manifest, args)
        if why:
            vid = manifest["script"]["passenger"]["voice_id"]
            bad_voices[vid] += 1
            rejected.append({"id": m.parent.name, "voice": vid, "reason": why})
            if args.quarantine:
                m.rename(m.with_name("manifest.rejected.json"))
    out = Path(args.turns_dir) / "_voice_rejects.json"
    out.write_text(json.dumps({"checked": n, "rejected": rejected, "by_voice": dict(bad_voices.most_common())}, indent=1))
    print(f"checked {n}, rejected {len(rejected)}; worst voices: {bad_voices.most_common(5)} -> {out}")


if __name__ == "__main__":
    main()
