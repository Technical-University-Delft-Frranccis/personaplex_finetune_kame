"""Dialogue-script schema, shared by tools/generate_scripts.py (LLM pod) and
tools/synthesize_turns.py (TTS env). Pure Python on purpose: no torch, so it also runs on a
llama.cpp pod. synthesize_turns.py can drop its own copy and use:

    from tools.script_schema import VALID_EMOTIONS, validate_script

The hard rules here are exactly the ones synthesize_turns.py enforces, so a script that passes
here will not be rejected later as INVALID.
"""

from __future__ import annotations

import re

VALID_SPEAKERS = ("passenger", "crew")
VALID_EMOTIONS = {
    "neutral", "irritated", "angry", "anxious", "distressed", "sad",
    "sulking", "embarrassed", "relieved", "grateful",
}
CALM_EMOTIONS = {"neutral", "relieved", "grateful"}
MAX_PASSENGER_WORDS = 30  # the oracle LLM is capped at 30 words (MAX_WORDS in generate_oracle_local.py)

_DIGIT = re.compile(r"\d")
_STAGE = re.compile(r"[\[\]\(\)\*]")


def validate_turns(turns: list[dict]) -> list[str]:
    """Hard rules (identical to synthesize_turns.validate_script)."""
    errors: list[str] = []
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


def validate_script(script: dict, voices: dict) -> list[str]:
    """Drop-in replacement for synthesize_turns.validate_script."""
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
    return errors + validate_turns(script["turns"])
