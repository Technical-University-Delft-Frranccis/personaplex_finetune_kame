"""Step 2 - turn per-turn TTS clips into full-duplex stereo dialogues in the KAME layout.

For each dialogue rendered by tools/synthesize_turns.py (Qwen3-TTS):
  1. trim TTS leading/trailing silence and apply a mild loudness step per passenger intensity
  2. word-align every clip against its known script text (tools/align_utils.py)
  3. place clips on a two-channel timeline: pauses (gap_s), backchannels over the other
     speaker, and interruptions that cut the interrupted speaker off mid-sentence
  4. loudness-normalize each channel, write the canonical files

Canonical speaker mapping (fixed for the whole pipeline):
  A = passenger = LEFT channel  = the model's own stream   (--moshi_speakers A)
  B = crew      = RIGHT channel = the trainee / user stream

Outputs:
  <out_dir>/audio/<id>.wav   stereo, 24 kHz, int16
  <out_dir>/text/<id>.json   [{"speaker","word","start","end"}]  (kame_finetune canonical)
  <out_dir>/meta/<id>.json   prompt text, voice prompt path, per-turn timing and emotion labels
                             (labels are unused by the context-control run, but keep them:
                             they are free training targets for a later emotion stream)

Runs in the kame_finetune env (torch/torchaudio 2.4.1 + soundfile). Optional: pyloudnorm.
    uv run -m tools.assemble_dialogues --turns_dir data/cabin/turns \
        --voices data/cabin/voices.json --out_dir data/cabin
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from tools.align_utils import WordAligner

SR = 24_000  # Mimi sample rate
ROLE_TO_SPK = {"passenger": "A", "crew": "B"}
# Extra loudness per passenger intensity. 0 by default: Qwen3-TTS emotion references already
# carry loudness; raise with --intensity_gain_db only if angry turns sound too quiet.
INTENSITY_STEPS = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}
TARGET_LUFS = -24.0  # PersonaPlex normalizes voice prompts to -24 LUFS


@dataclass
class Placement:
    idx: int
    spk: str
    clip: torch.Tensor           # [T] mono @ SR
    words: list                  # AlignedWord, clip-relative
    start: float = 0.0
    cut_at: float | None = None  # absolute time the speaker is cut off (interrupted)
    backchannel: bool = False
    turn: dict = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return self.clip.numel() / SR

    @property
    def end(self) -> float:
        return self.cut_at if self.cut_at is not None else self.start + self.dur


def trim_silence(wav: torch.Tensor, thresh_db: float = -45.0, pad_s: float = 0.04) -> torch.Tensor:
    frame = int(0.01 * SR)
    if wav.numel() < frame * 3:
        return wav
    frames = wav[: wav.numel() // frame * frame].view(-1, frame)
    level = 20 * torch.log10(frames.abs().amax(dim=1).clamp_min(1e-6))
    active = torch.nonzero(level > thresh_db).flatten()
    if active.numel() == 0:
        return wav
    pad = int(pad_s * SR)
    lo = max(0, int(active[0]) * frame - pad)
    hi = min(wav.numel(), (int(active[-1]) + 1) * frame + pad)
    return wav[lo:hi]


def load_clip(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def quality_ok(turn_text: str, words: list, dur: float) -> str | None:
    n_script = len([w for w in turn_text.split() if any(c.isalpha() for c in w)])
    if n_script == 0:
        return "no words"
    if len(words) < 0.9 * n_script:
        return f"aligned {len(words)}/{n_script} words"
    if dur > 1.2 + 1.0 * n_script:  # > ~1 s per word: TTS rambled or hung
        return f"clip too long ({dur:.1f}s for {n_script} words)"
    mean_score = sum(w.score for w in words) / max(1, len(words))
    if mean_score < 0.35:
        return f"low alignment confidence {mean_score:.2f} (TTS mispronounced or skipped words)"
    return None


def place_turns(placements: list[Placement], rng: random.Random, lead_in: float = 0.4) -> None:
    """Assign absolute start times; mark interrupted turns with cut_at."""
    floor: Placement | None = None          # last non-backchannel turn
    chan_end = {"A": 0.0, "B": 0.0}          # a speaker never overlaps themselves
    for p in placements:
        t = p.turn
        if floor is None:
            start = lead_in
        elif p.backchannel:
            start = floor.start + rng.uniform(0.35, 0.75) * floor.dur
        elif t.get("interrupts") and floor.spk != p.spk and floor.dur >= 1.2:
            start = floor.start + max(0.6, rng.uniform(0.45, 0.8) * floor.dur)
            floor.cut_at = min(floor.start + floor.dur, start + rng.uniform(0.25, 0.5))
            chan_end[floor.spk] = floor.cut_at
        else:
            gap = t.get("gap_s")
            gap = rng.uniform(0.2, 0.8) if gap is None else float(gap)
            if floor.spk == p.spk:
                gap = max(gap, 0.15)
            start = floor.end + gap
        start = max(start, chan_end[p.spk] + 0.05)
        p.start = round(start, 3)
        chan_end[p.spk] = p.start + p.dur
        if not p.backchannel:
            floor = p


def render(placements: list[Placement], tail: float = 0.6) -> np.ndarray:
    total = max(p.end for p in placements) + tail
    out = torch.zeros(2, int(total * SR) + 1)
    fade = int(0.04 * SR)
    for p in placements:
        seg = p.clip
        if p.cut_at is not None:
            keep = max(fade, int((p.cut_at - p.start) * SR))
            seg = seg[:keep].clone()
            seg[-fade:] *= torch.linspace(1.0, 0.0, fade)
        ch = 0 if p.spk == "A" else 1
        s = int(p.start * SR)
        out[ch, s : s + seg.numel()] += seg
    return out.numpy()


def normalize_channel(x: np.ndarray) -> np.ndarray:
    """Whole-channel loudness normalization keeps loud (angry) turns louder than quiet ones."""
    if np.abs(x).max() < 1e-5:
        return x
    try:
        import pyloudnorm as pyln

        loud = pyln.Meter(SR).integrated_loudness(x)
        x = pyln.normalize.loudness(x, loud, TARGET_LUFS)
    except ImportError:
        rms = np.sqrt(np.mean(x[np.abs(x) > 1e-4] ** 2))
        x = x * (10 ** (TARGET_LUFS / 20) / max(rms, 1e-6))
    peak = np.abs(x).max()
    if peak > 0.89:  # -1 dBFS ceiling; hot input clips inside Mimi
        x = x * (0.89 / peak)
    return x


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    off = rng.randint(0, len(noise) - len(x))
    n = noise[off : off + len(x)]
    sig = np.sqrt(np.mean(x[np.abs(x) > 1e-4] ** 2)) if np.any(np.abs(x) > 1e-4) else 1e-3
    n = n * (sig / (np.sqrt(np.mean(n**2)) + 1e-9)) * 10 ** (-snr_db / 20)
    return x + n


def process(dlg_dir: Path, aligner: WordAligner, voices: dict, out_dir: Path,
            noise: np.ndarray | None, snr_range: tuple[float, float], gain_db: float = 0.0) -> str:
    manifest = json.loads((dlg_dir / "manifest.json").read_text())
    script = manifest["script"]
    did = script["id"]
    if (out_dir / "meta" / f"{did}.json").exists():
        return "skipped (done)"
    rng = random.Random(did)

    placements: list[Placement] = []
    for i, (turn, clip_name) in enumerate(zip(script["turns"], manifest["clips"], strict=True)):
        spk = ROLE_TO_SPK[turn["speaker"]]
        clip = trim_silence(load_clip(dlg_dir / clip_name))
        words = aligner.align(clip[None], SR, turn["text"])
        bad = quality_ok(turn["text"], words, clip.numel() / SR)
        if bad:
            return f"REJECTED turn {i}: {bad}"
        if spk == "A":
            clip = clip * 10 ** (gain_db * INTENSITY_STEPS.get(turn.get("intensity", 0), 0.0) / 20)
        placements.append(Placement(i, spk, clip, words, backchannel=bool(turn.get("backchannel")), turn=turn))

    place_turns(placements, rng)
    audio = render(placements)
    if noise is not None:
        audio[1] = add_noise(audio[1], noise, rng.uniform(*snr_range), rng)
    audio[0] = normalize_channel(audio[0])
    audio[1] = normalize_channel(audio[1])

    words_out, turns_meta = [], []
    for p in placements:
        kept = [w for w in p.words if p.cut_at is None or p.start + w.end <= p.cut_at]
        for w in kept:
            words_out.append({"speaker": p.spk, "word": w.word,
                              "start": round(p.start + w.start, 3), "end": round(p.start + w.end, 3)})
        turns_meta.append({
            "speaker": p.spk, "text": p.turn["text"], "spoken_text": " ".join(w.word for w in kept),
            "start": p.start, "end": round(p.end, 3), "interrupted": p.cut_at is not None,
            "backchannel": p.backchannel, "emotion": p.turn.get("emotion"),
            "intensity": p.turn.get("intensity"), "trigger": p.turn.get("trigger"),
            "tts_ref": manifest.get("refs", {}).get(str(p.idx)),
            "speaker_similarity": manifest.get("speaker_similarity", {}).get(str(p.idx)),
        })
    words_out.sort(key=lambda w: (w["start"], w["speaker"]))

    for sub in ("audio", "text", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "audio" / f"{did}.wav", audio.T, SR, subtype="PCM_16")
    (out_dir / "text" / f"{did}.json").write_text(json.dumps(words_out, ensure_ascii=False))
    meta = {
        "id": did,
        "prompt_text": script["passenger"]["brief"],
        # the exact file PersonaPlex gets as voice prompt at inference (make the .pt from it)
        "voice_prompt_wav": voices[script["passenger"]["voice_id"]]["prompt_wav"],
        "passenger_voice_id": script["passenger"]["voice_id"],
        "crew_voice_id": script["crew"]["voice_id"],
        "tts": manifest.get("tts"),
        "scenario": script.get("scenario", {}),
        "crew": script.get("crew", {}),
        "duration_s": round(audio.shape[1] / SR, 2),
        "turns": turns_meta,
    }
    (out_dir / "meta" / f"{did}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return f"ok {meta['duration_s']:.0f}s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--turns_dir", required=True)
    p.add_argument("--voices", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--noise_wav", default=None, help="optional cabin noise, mixed into the crew channel only")
    p.add_argument("--snr_min", type=float, default=15.0)
    p.add_argument("--snr_max", type=float, default=30.0)
    p.add_argument("--intensity_gain_db", type=float, default=0.0,
                   help="dB added per passenger intensity step (0 = off)")
    args = p.parse_args()

    voices = json.loads(Path(args.voices).read_text())
    noise = None
    if args.noise_wav:
        n, nsr = torchaudio.load(args.noise_wav)
        noise = torchaudio.functional.resample(n.mean(0), nsr, SR).numpy()

    aligner = WordAligner(args.device)
    out_dir = Path(args.out_dir)
    dirs = sorted(d for d in Path(args.turns_dir).iterdir() if (d / "manifest.json").exists())
    n_ok = n_rej = 0
    for k, d in enumerate(dirs, 1):
        try:
            status = process(d, aligner, voices, out_dir, noise, (args.snr_min, args.snr_max),
                             args.intensity_gain_db)
        except Exception as e:
            status = f"ERROR {type(e).__name__}: {e}"
        n_ok += status.startswith("ok")
        n_rej += status.startswith(("REJECTED", "ERROR"))
        print(f"[{k}/{len(dirs)}] {d.name}: {status}", flush=True)
    print(f"assembled {n_ok}, rejected {n_rej} (re-render rejected ones with another TTS seed)")


if __name__ == "__main__":
    main()
