"""Step 1 - render every script turn with Qwen3-TTS (TTS environment).

Each turn is cloned (Qwen3-TTS Base) from the voice-bank reference that matches its emotion
and intensity (see build_voice_bank.py), so identity comes from the voice's anchor and
delivery from the emotional reference + the text. Every rendered turn is checked against the
voice's anchor with a speaker verifier and re-sampled up to --retries times if it drifted.

Output (unchanged layout, consumed by assemble_dialogues.py):
    <out_dir>/<dialogue_id>/000.wav, 001.wav, ...
    <out_dir>/<dialogue_id>/manifest.json   (written last = "dialogue complete" marker)

    python -m tools.synthesize_turns --scripts_dir data/cabin/scripts \
        --voices data/cabin/voices.json --out_dir data/cabin/turns --device cuda:0 --shard 0 --num_shards 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

from tools.speaker_verify import SpeakerVerifier

VALID_SPEAKERS = ("passenger", "crew")
VALID_EMOTIONS = {
    "neutral", "irritated", "angry", "anxious", "distressed", "sad",
    "sulking", "embarrassed", "relieved", "grateful",
}
_DIGIT = re.compile(r"\d")
_STAGE = re.compile(r"[\[\]\(\)\*]")


def pick_ref(voice: dict, emotion: str | None, intensity: int) -> str:
    """Reference key for a turn: exact emotion_intensity, else the closest intensity of the
    same emotion, else neutral. Crew and pool voices only have 'neutral'."""
    refs = voice["refs"]
    if not emotion or emotion == "neutral" or intensity <= 0:
        return "neutral"
    same = [(abs(int(k.rsplit("_", 1)[1]) - intensity), k) for k in refs
            if k != "neutral" and k.rsplit("_", 1)[0] == emotion]
    return min(same)[1] if same else "neutral"


def max_tokens_for(text: str) -> int:
    """Cap generation length (12.5 codec frames/s): ~1 s per word + 3 s, so a runaway
    generation is cut instead of producing a minute of babble."""
    return int(12.5 * (1.0 * len(text.split()) + 3.0))


def validate_script(script: dict, voices: dict) -> list[str]:
    errors: list[str] = []
    for key in ("id", "passenger", "crew", "turns"):
        if key not in script:
            errors.append(f"missing key {key!r}")
    if errors:
        return errors
    for role in ("passenger", "crew"):
        vid = script[role].get("voice_id")
        if vid not in voices:
            errors.append(f"{role}.voice_id {vid!r} not in voices.json")
        elif voices[vid]["role"] != role:
            errors.append(f"{role}.voice_id {vid!r} is registered as a {voices[vid]['role']} voice")
    if not script["passenger"].get("brief", "").strip():
        errors.append("passenger.brief is empty")
    turns = script["turns"]
    if len(turns) < 4:
        errors.append("fewer than 4 turns")
    floor = None
    for i, t in enumerate(turns):
        sp, text = t.get("speaker"), (t.get("text") or "").strip()
        if sp not in VALID_SPEAKERS:
            errors.append(f"turn {i}: bad speaker {sp!r}")
            continue
        if not text:
            errors.append(f"turn {i}: empty text")
        if _DIGIT.search(text):
            errors.append(f"turn {i}: digits in text (write numbers as words)")
        if _STAGE.search(text):
            errors.append(f"turn {i}: brackets/asterisks in text")
        if t.get("backchannel") and t.get("interrupts"):
            errors.append(f"turn {i}: both backchannel and interrupt")
        if i == 0 and (t.get("backchannel") or t.get("interrupts")):
            errors.append("turn 0: cannot be a backchannel/interrupt")
        if (t.get("backchannel") or t.get("interrupts")) and sp == floor:
            errors.append(f"turn {i}: backchannel/interrupt by the speaker holding the floor")
        if sp == "passenger":
            if t.get("emotion") not in VALID_EMOTIONS:
                errors.append(f"turn {i}: emotion {t.get('emotion')!r} not in label set")
            if not isinstance(t.get("intensity"), int) or not 0 <= t["intensity"] <= 3:
                errors.append(f"turn {i}: intensity must be int 0-3")
        if not t.get("backchannel"):
            floor = sp
    return errors


class QwenCloneTTS:
    def __init__(self, voices: dict, device: str, attn: str, model_id: str, batch_size: int):
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=torch.bfloat16,
                                                   attn_implementation=attn)
        self.voices = voices
        self.batch_size = batch_size
        self._prompts: dict[tuple[str, str], object] = {}

    def prompt(self, voice_id: str, ref_key: str):
        key = (voice_id, ref_key)
        if key not in self._prompts:
            ref = self.voices[voice_id]["refs"][ref_key]
            self._prompts[key] = self.model.create_voice_clone_prompt(ref_audio=ref["wav"], ref_text=ref["text"])
        return self._prompts[key]

    def render(self, voice_id: str, ref_key: str, texts: list[str]):
        """Batched generation for turns that share one reference. Returns (wavs, sr)."""
        out, sr = [], 24000
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            wavs, sr = self.model.generate_voice_clone(
                text=chunk, language=["English"] * len(chunk),
                voice_clone_prompt=self.prompt(voice_id, ref_key),
                max_new_tokens=max(max_tokens_for(t) for t in chunk),
            )
            out.extend(wavs)
        return out, sr


def process_script(path: Path, tts: QwenCloneTTS, sv: SpeakerVerifier, anchors: dict, args) -> str:
    script = json.loads(path.read_text())
    did = script.get("id", path.stem)
    ddir = Path(args.out_dir) / did
    if (ddir / "manifest.json").exists():
        return "skipped (done)"
    errors = validate_script(script, tts.voices)
    if errors:
        return "INVALID: " + "; ".join(errors[:5])
    ddir.mkdir(parents=True, exist_ok=True)

    # plan: which voice + reference each turn uses, grouped for batching
    plan, groups = {}, defaultdict(list)
    for i, t in enumerate(script["turns"]):
        vid = script[t["speaker"]]["voice_id"]
        ref_key = pick_ref(tts.voices[vid], t.get("emotion"), int(t.get("intensity", 0)))
        plan[i] = (vid, ref_key)
        if not (ddir / f"{i:03d}.wav").exists():
            groups[(vid, ref_key)].append(i)

    sims: dict[int, float] = {}
    for (vid, ref_key), idxs in groups.items():
        pending = list(idxs)
        for attempt in range(args.retries + 1):
            if not pending:
                break
            wavs, sr = tts.render(vid, ref_key, [script["turns"][i]["text"].strip() for i in pending])
            still = []
            for i, wav in zip(pending, wavs, strict=True):
                sim = SpeakerVerifier.similarity(anchors[vid], sv.embed(wav, sr))
                if sim >= args.min_similarity or attempt == args.retries:
                    sims[i] = round(sim, 3)
                    sf.write(ddir / f"{i:03d}.wav", wav, sr)
                else:
                    still.append(i)
            pending = still
        low = [i for i in idxs if sims.get(i, 1.0) < args.min_similarity]
        if low:
            for i in low:  # delete so a rerun re-renders them instead of silently accepting
                (ddir / f"{i:03d}.wav").unlink(missing_ok=True)
            return f"REJECTED: turns {low} drifted from voice {vid} (sim < {args.min_similarity})"

    manifest = {
        "script": script,
        "clips": [f"{i:03d}.wav" for i in range(len(script["turns"]))],
        "refs": {str(i): plan[i][1] for i in plan},
        "speaker_similarity": {str(i): s for i, s in sims.items()},
        "tts": args.model_id,
    }
    tmp = ddir / "manifest.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    tmp.replace(ddir / "manifest.json")
    return f"ok ({len(plan)} turns, {len(sims)} rendered)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scripts_dir", required=True)
    p.add_argument("--voices", required=True, help="registry written by build_voice_bank.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn", default="sdpa", help="flash_attention_2 if installed")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--min_similarity", type=float, default=0.75, help="same threshold idea as the voice bank")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    voices = json.loads(Path(args.voices).read_text())
    scripts = sorted(Path(args.scripts_dir).glob("*.json"))
    scripts = [s for i, s in enumerate(scripts) if i % args.num_shards == args.shard]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tts = QwenCloneTTS(voices, args.device, args.attn, args.model_id, args.batch_size)
    sv = SpeakerVerifier("cuda" if args.device.startswith("cuda") else args.device)
    anchors = {vid: sv.embed_file(v["refs"]["neutral"]["wav"]) for vid, v in voices.items()}

    t0, bad = time.time(), 0
    for k, path in enumerate(scripts, 1):
        try:
            status = process_script(path, tts, sv, anchors, args)
        except Exception as e:
            status = f"ERROR {type(e).__name__}: {e}"
        bad += not status.startswith(("ok", "skipped"))
        print(f"[{k}/{len(scripts)}] {path.stem}: {status} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done, {bad} failed/invalid/rejected", file=sys.stderr)


if __name__ == "__main__":
    main()
