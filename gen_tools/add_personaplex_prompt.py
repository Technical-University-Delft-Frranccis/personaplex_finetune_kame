"""Step 5 - prepend PersonaPlex's hybrid system prompt to every training row.

Run after tools.prepare_dataset. Reads its parquet files and writes new ones in which the
passenger stream (A) starts exactly like a PersonaPlex inference session does:

    phase         frames      agent text (A)        agent audio (A)        user audio (B)
    voice prompt  Tv - 1      PAD                   Mimi(voice prompt)     sine
    silence       S = 6       PAD                   silence tokens         sine
    text prompt   Lt          <system> brief <system> silence tokens       sine
    silence       S = 6       PAD                   silence tokens         sine
    conversation  T           (unchanged)

PersonaPlex's LMGen drops the first voice-prompt frame (its offset-0 step overwrites it with
the initial token), so the prefix starts at voice frame 1 to match inference exactly.

Also:
  * shifts oracle events by the prefix length and drops events earlier than
    --min_event_frame (so KAME's left-shift augmentation, up to 15 frames, can never move
    oracle tokens into the prompt)
  * truncates rows to --max_frames INCLUDING the prefix (rows must never be split by
    --max_length, or only the first chunk would carry the prompt)
  * adds A_prompt_frames / B_prompt_frames columns; utils/data.py masks the loss there

    uv run -m tools.add_personaplex_prompt \
        --parquet_glob 'processed_data/cabin/train-*.parquet' --meta_dir data/cabin/meta \
        --output_prefix processed_data/cabin_pp/train --max_frames 2400
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# From PersonaPlex moshi/models/lm.py - Mimi codes of a silent frame and of the 440 Hz sine.
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)
TEXT_PAD = 3
FRAME_RATE = 12.5
SILENCE_FRAMES = int(0.5 * FRAME_RATE)  # PersonaPlex: audio_silence_frame_cnt=int(0.5 * frame_rate)
ORACLE_FIELDS = ("event_frame_pos", "event_ratio", "event_skip_forbid",
                 "pred_values", "pred_offsets", "hint_values", "hint_offsets")


def wrap_with_system_tags(text: str) -> str:
    """Identical to PersonaPlex server.py / offline.py."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


# ----------------------------------------------------------------------------- pure numpy core
def subset_ragged(values: np.ndarray, offsets: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    new_offsets = np.zeros(len(keep) + 1, dtype=np.int32)
    pieces = []
    for j, i in enumerate(keep):
        seg = values[offsets[i] : offsets[i + 1]]
        pieces.append(seg)
        new_offsets[j + 1] = new_offsets[j] + len(seg)
    new_values = np.concatenate(pieces).astype(np.int32) if pieces else np.zeros(0, dtype=np.int32)
    return new_values, new_offsets


def build_prefix(voice_codes: np.ndarray, text_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (agent_prefix [9, P], user_prefix [9, P]) in the layout of prepare_dataset rows."""
    voice = voice_codes[:, 1:]  # LMGen drops voice frame 0 (see module docstring)
    tv, lt, s = voice.shape[1], len(text_ids), SILENCE_FRAMES
    p = tv + s + lt + s
    agent_text = np.array([TEXT_PAD] * (tv + s) + list(text_ids) + [TEXT_PAD] * s, dtype=np.int64)
    agent_audio = np.concatenate([voice, np.tile(SILENCE_TOKENS[:, None], (1, s + lt + s))], axis=1)
    agent = np.concatenate([agent_text[None], agent_audio], axis=0)
    user = np.concatenate([np.full((1, p), TEXT_PAD), np.tile(SINE_TOKENS[:, None], (1, p))], axis=0)
    assert agent.shape == user.shape == (9, p)
    return agent, user


def prompt_row(row: dict, agent_prefix: np.ndarray, user_prefix: np.ndarray,
               max_frames: int, min_event_frame: int, min_conv_frames: int) -> dict | None:
    """Prepend the prefix to one prepare_dataset row. Speaker A must be the passenger/model."""
    a = np.stack([np.asarray(x) for x in row["A"]])
    b = np.stack([np.asarray(x) for x in row["B"]])
    p = agent_prefix.shape[1]
    t_keep = min(a.shape[1], max_frames - p)
    if t_keep < min_conv_frames:
        return None

    out = dict(row)
    out["A"] = np.concatenate([agent_prefix, a[:, :t_keep]], axis=1).astype(np.int32).tolist()
    out["B"] = np.concatenate([user_prefix, b[:, :t_keep]], axis=1).astype(np.int32).tolist()
    out["A_prompt_frames"] = int(p)
    out["B_prompt_frames"] = int(p)  # B is never trained as main speaker; masked for safety

    for sp in ("A", "B"):
        base = f"{sp}_oracle_"
        if base + "event_frame_pos" not in row:
            continue
        pos = np.asarray(row[base + "event_frame_pos"], dtype=np.int64)
        keep = np.nonzero((pos >= min_event_frame) & (pos < t_keep))[0]
        out[base + "event_frame_pos"] = (pos[keep] + p).astype(np.int32).tolist()
        out[base + "event_ratio"] = np.asarray(row[base + "event_ratio"], dtype=np.float32)[keep].tolist()
        out[base + "event_skip_forbid"] = np.asarray(row[base + "event_skip_forbid"], dtype=np.int8)[keep].tolist()
        for kind in ("pred", "hint"):
            v, o = subset_ragged(np.asarray(row[base + f"{kind}_values"], dtype=np.int32),
                                 np.asarray(row[base + f"{kind}_offsets"], dtype=np.int64), keep)
            out[base + f"{kind}_values"] = v.tolist()
            out[base + f"{kind}_offsets"] = o.tolist()
    return out


# ----------------------------------------------------------------------------- encoders
class PromptEncoder:
    def __init__(self, repo: str, device: str, voice_prompt_max_s: float):
        import sentencepiece
        import torch
        from huggingface_hub import hf_hub_download
        from kame.models import loaders

        self.torch = torch
        self.device = device
        self.max_s = voice_prompt_max_s
        self.mimi = loaders.get_mimi(filename=hf_hub_download(repo, loaders.MIMI_NAME), device=device)
        self.sp = sentencepiece.SentencePieceProcessor(
            model_file=hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME))
        self._voice_cache: dict[str, np.ndarray] = {}

    def text_ids(self, brief: str) -> list[int]:
        return list(self.sp.encode(wrap_with_system_tags(brief)))

    def voice_codes(self, wav_path: str) -> np.ndarray:
        if wav_path in self._voice_cache:
            return self._voice_cache[wav_path]
        import torchaudio

        torch = self.torch
        wav, sr = torchaudio.load(wav_path)
        wav = torchaudio.functional.resample(wav.mean(0), sr, self.mimi.sample_rate)
        if self.max_s > 0:  # default 0: use the whole clip, exactly like PersonaPlex at inference
            wav = wav[: int(self.max_s * self.mimi.sample_rate)]
        wav = wav.numpy().astype(np.float64)
        try:  # PersonaPlex normalizes voice prompts to -24 LUFS before encoding
            import pyloudnorm as pyln

            wav = pyln.normalize.loudness(wav, pyln.Meter(self.mimi.sample_rate).integrated_loudness(wav), -24.0)
        except ImportError:
            print("WARNING: pyloudnorm missing; voice prompt not LUFS-normalized like PersonaPlex")
        frame = int(self.mimi.sample_rate / self.mimi.frame_rate)
        n = math.ceil(len(wav) / frame) * frame
        wav = np.pad(wav, (0, n - len(wav)))  # PersonaPlex zero-pads the last frame
        x = torch.from_numpy(wav).float().view(1, 1, -1).to(self.device)
        with torch.no_grad():
            codes = self.mimi.encode(x)[0].cpu().numpy().astype(np.int64)  # [8, Tv]
        self._voice_cache[wav_path] = codes
        return codes


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet_glob", required=True)
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--output_prefix", required=True)
    ap.add_argument("--repo", default="nvidia/personaplex-7b-v1", help="source of Mimi + text tokenizer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=2400, help="prefix + conversation; keep < --max_length")
    ap.add_argument("--min_event_frame", type=int, default=16, help="> KAME's max left shift (15)")
    ap.add_argument("--min_conv_frames", type=int, default=250, help="drop rows with < 20 s of dialogue")
    ap.add_argument("--voice_prompt_max_s", type=float, default=0.0,
                    help="0 = full clip (matches inference). Keep prompt.wav files short instead.")
    args = ap.parse_args()

    enc = PromptEncoder(args.repo, args.device, args.voice_prompt_max_s)
    files = sorted(glob.glob(args.parquet_glob))
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    total_in = total_out = truncated = 0
    for k, path in enumerate(files, 1):
        df = pd.read_parquet(path)
        rows = []
        for row in df.to_dict(orient="records"):
            total_in += 1
            name = os.path.basename(row["dialogue_id"])
            meta = json.loads((Path(args.meta_dir) / f"{name}.json").read_text())
            agent, user = build_prefix(enc.voice_codes(meta["voice_prompt_wav"]), enc.text_ids(meta["prompt_text"]))
            new = prompt_row(row, agent, user, args.max_frames, args.min_event_frame, args.min_conv_frames)
            if new is None:
                print(f"  dropped {name}: too short after prefix")
                continue
            truncated += len(row["A"][0]) + agent.shape[1] > args.max_frames
            rows.append(new)
        out = f"{args.output_prefix}-{k:03d}-of-{len(files):03d}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        total_out += len(rows)
        print(f"[{k}/{len(files)}] {path} -> {out} ({len(rows)} rows)")
    print(f"rows in {total_in}, out {total_out}, truncated {truncated}. "
          f"Train with --max_length >= {args.max_frames + 2} so no row is ever split.")


if __name__ == "__main__":
    main()
