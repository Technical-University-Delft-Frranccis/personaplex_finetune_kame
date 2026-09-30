"""Step 0 - build the voice bank with Qwen3-TTS (TTS environment).

Why this step exists: the open Qwen3-TTS models split the two things we need.
  * VoiceDesign  - makes a voice from a text description AND follows emotion instructions,
                   but every call is a new, slightly different speaker
  * Base (clone) - keeps a speaker's identity from a reference clip, but takes NO instruction;
                   delivery follows the reference clip and the text
So we (1) design each voice once and fix an identity anchor, then (2) create one reference
clip per emotion/intensity that is verified to be the SAME speaker as the anchor.
synthesize_turns.py later clones from the emotion-matched reference.

For each voice in the spec file:
  1. neutral anchor: N VoiceDesign candidates of a neutral paragraph; the medoid (most typical)
     becomes neutral.wav, and also prompt.wav = the PersonaPlex voice prompt used in training
     AND at inference (make the .pt for custom_voices/ from this exact file)
  2. hero passengers only: for every emotion level, K VoiceDesign candidates with
     "<description>. <emotion direction>"; keep the one most similar to the anchor if it
     passes --min_similarity, else fall back to a Base clone of the anchor speaking the
     emotional line (identity guaranteed, emotion weaker; marked source="clone_fallback")

Spec (JSON list):
  [{"voice_id": "pax_f_01", "role": "passenger", "tier": "hero",
    "description": "Female, about 45, British English, low alto, slightly husky, speaks fast"},
   {"voice_id": "crew_m_03", "role": "crew", "tier": "pool",
    "description": "Male, about 30, Dutch-accented English, calm tenor"}]

    python -m tools.build_voice_bank --spec voice_specs.json --out_dir data/cabin/voices \
        --registry data/cabin/voices.json
Listen to data/cabin/voices/<id>/ before generating dialogues; scores are in the registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tools.speaker_verify import SpeakerVerifier, medoid_index

NEUTRAL_TEXT = ("Good evening. I think this is my seat, twelve C, next to the window. "
                "It has been a long day, so I am just glad to finally sit down.")

# (emotion, intensity) -> (direction appended to the voice description, line to speak)
EMOTION_LEVELS: dict[tuple[str, int], tuple[str, str]] = {
    ("irritated", 1): ("Mildly irritated, clipped and impatient.",
                       "Excuse me. I asked for water twenty minutes ago. Is it still coming?"),
    ("irritated", 2): ("Clearly irritated, sharp tone, sighing between phrases.",
                       "No, that is not what I asked. I have asked three times now."),
    ("angry", 2): ("Angry, raised voice, fast and forceful.",
                   "You cannot just tell me to sit down. I paid for this seat like everybody else."),
    ("angry", 3): ("Very angry, loud and sharp, almost shouting but not screaming.",
                   "Do not talk to me like that! Everyone is looking at me now, thanks a lot!"),
    ("anxious", 1): ("Slightly anxious, quick breathing, uncertain.",
                     "Sorry, is that noise normal? The wing is making a strange sound."),
    ("anxious", 2): ("Very anxious, tense and fast, voice trembling a little.",
                     "I really need to know what is happening. Are we going to be okay?"),
    ("distressed", 2): ("Distressed, voice breaking, struggling to stay composed.",
                        "My daughter was supposed to be on this flight. I cannot reach her. I just, I cannot."),
    ("sad", 1): ("Quietly sad, slow and low.",
                 "I am flying home for my father's funeral. I just want this flight to be over."),
    ("sulking", 1): ("Sulking, flat and short, reluctant.",
                     "Fine. Whatever. Just bring the sandwich then."),
    ("embarrassed", 1): ("Embarrassed, softer and hesitant, trailing off.",
                         "Oh. Right. Sorry, I did not realise everyone could hear me."),
    ("relieved", 1): ("Relieved, exhaling, warmer tone.",
                      "Oh thank goodness. Okay. Thank you, that really helps."),
    ("grateful", 1): ("Warm and grateful, sincere.",
                      "Thank you so much for sorting that out, I really appreciate it."),
}


def load_models(device: str, attn: str, need_design: bool):
    from qwen_tts import Qwen3TTSModel

    kw = dict(device_map=device, dtype=torch.bfloat16, attn_implementation=attn)
    design = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", **kw) if need_design else None
    clone = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", **kw)
    return design, clone


def design_candidates(design, text: str, instruct: str, n: int) -> tuple[list[np.ndarray], int]:
    wavs, sr = design.generate_voice_design(
        text=[text] * n, language=["English"] * n, instruct=[instruct] * n)
    return list(wavs), sr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--registry", required=True, help="voices.json read by the rest of the pipeline")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn", default="sdpa", help="flash_attention_2 if installed")
    p.add_argument("--neutral_candidates", type=int, default=6)
    p.add_argument("--emotion_candidates", type=int, default=6)
    p.add_argument("--min_similarity", type=float, default=0.80, help="tune on a listening test")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    specs = json.loads(Path(args.spec).read_text())
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    design, clone = load_models(args.device, args.attn, need_design=True)
    sv = SpeakerVerifier(args.device.split(":")[0] if args.device.startswith("cuda") else args.device)

    for spec in specs:
        vid = spec["voice_id"]
        if vid in registry:
            print(f"{vid}: already in registry, skipped (delete the entry to rebuild)")
            continue
        vdir = Path(args.out_dir) / vid
        vdir.mkdir(parents=True, exist_ok=True)
        desc = spec["description"].strip().rstrip(".")

        # 1. neutral anchor = medoid of N designed candidates
        cands, sr = design_candidates(design, NEUTRAL_TEXT, f"{desc}. Calm, neutral, natural delivery.",
                                      args.neutral_candidates)
        embs = [sv.embed(w, sr) for w in cands]
        k = medoid_index(embs)
        anchor = embs[k]
        for i, w in enumerate(cands):
            sf.write(vdir / f"neutral_cand{i}.wav", w, sr)
        sf.write(vdir / "neutral.wav", cands[k], sr)
        sf.write(vdir / "prompt.wav", cands[k], sr)
        dur = len(cands[k]) / sr
        entry = {
            "role": spec["role"], "tier": spec.get("tier", "pool"), "description": spec["description"],
            "prompt_wav": str((vdir / "prompt.wav").resolve()),
            "refs": {"neutral": {"wav": str((vdir / "neutral.wav").resolve()), "text": NEUTRAL_TEXT,
                                 "similarity": 1.0, "source": "design_medoid"}},
        }
        print(f"{vid}: anchor = candidate {k} ({dur:.1f}s)" + ("  WARNING >10 s" if dur > 10 else ""))

        # 2. emotion references (hero passengers only)
        if spec["role"] == "passenger" and spec.get("tier") == "hero":
            anchor_prompt = clone.create_voice_clone_prompt(
                ref_audio=str(vdir / "neutral.wav"), ref_text=NEUTRAL_TEXT)
            for (emotion, level), (direction, line) in EMOTION_LEVELS.items():
                key = f"{emotion}_{level}"
                cands, sr = design_candidates(design, line, f"{desc}. {direction}", args.emotion_candidates)
                sims = [SpeakerVerifier.similarity(anchor, sv.embed(w, sr)) for w in cands]
                best = int(np.argmax(sims))
                if sims[best] >= args.min_similarity:
                    wav, sim, source = cands[best], sims[best], "design"
                else:
                    wavs, sr = clone.generate_voice_clone(text=line, language="English",
                                                          voice_clone_prompt=anchor_prompt)
                    wav, source = wavs[0], "clone_fallback"
                    sim = SpeakerVerifier.similarity(anchor, sv.embed(wav, sr))
                path = vdir / f"{key}.wav"
                sf.write(path, wav, sr)
                entry["refs"][key] = {"wav": str(path.resolve()), "text": line,
                                      "similarity": round(sim, 3), "source": source}
                print(f"  {key:14s} sim={sim:.3f} ({source}; best design candidate {max(sims):.3f})")

        registry[vid] = entry
        registry_path.write_text(json.dumps(registry, indent=1))
    print(f"registry -> {registry_path}")


if __name__ == "__main__":
    main()
