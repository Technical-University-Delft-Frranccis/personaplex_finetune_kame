"""Step -1 - generate dialogue scripts with the local LLM (llama-server / vLLM, OpenAI API).

Pipeline position:
    generate_scripts.py -> synthesize_turns.py -> assemble_dialogues.py -> generate_oracle_local.py

For every dialogue index i (deterministic per --seed, so reruns resume cleanly):
  1. sample a seed from tools/scenario_taxonomy.py: group (conflict / distress / neutral / positive),
     category, phase, flight, cabin, crew performance + mistake, outcome, profanity, opener
  2. pick the passenger voice FIRST and give its description + a matching name to the LLM, so the
     brief's gender, age and background fit the voice (no "You are Mark" on a female voice)
       - conflict / distress: hero voices only (they have emotion references in the voice bank;
         pool voices would render every angry line from the neutral reference)
       - neutral / positive: hero and pool voices, so hero voices are also heard calm and the model
         does not tie a voice's identity to one emotion
  3. fill prompts/script_generation_prompt.md, call the LLM, parse JSON
  4. auto-fix overlap markup the assembler cannot use, then hard-validate (script_schema) and lint
     (passenger <= 30 words, no assistant phrases, triggers, outcome) - retry on failure
  5. write <out_dir>/<id>.json atomically; failures go to <out_dir>/_failed.jsonl

    python -m tools.generate_scripts --n 600 --voices voice_specs.json \
        --prompt prompts/script_generation_prompt.md --out_dir data/cabin/scripts \
        --base_url http://127.0.0.1:8080/v1 --model local --workers 8

--voices accepts the spec list (voice_specs.json) or the built registry (voices.json); both have
role / tier / description. Use --dry_run to print seeds and one filled prompt without the LLM.
Only needs: pip install openai
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools import scenario_taxonomy as tax
from tools.script_schema import CALM_EMOTIONS, MAX_PASSENGER_WORDS, VALID_EMOTIONS, validate_turns
from tools.spoken_numbers import numbers_to_words

# Phrases that make the passenger sound like an assistant or a sitcom. Lower-case substrings.
BANNED_PASSENGER = [
    "how can i help", "is there anything else", "i'd be happy to", "i would be happy to",
    "i understand your frustration", "i feel frustrated", "i feel unheard",
    "paid for this ticket", "like everyone else", "thanks a lot", "this is unacceptable",
    "do you know who i am", "have a great day", "let me know if",
]

# Names: ONE pool of short, internationally common first names with regular spelling. Spoken names
# go through TTS and the forced aligner; irregular names (Siobhan, Aoife, Thijs, Ngozi) get
# mispronounced, fail the aligner's confidence check in assemble_dialogues.py and waste the render.
# Add rarer names only after a listening test.
NAME_POOLS = {
    "intl": (["Anna", "Maria", "Sara", "Nina", "Eva", "Emma", "Julia", "Sofia", "Laura", "Elena", "Lena", "Mia",
              "Clara", "Nora", "Helen", "Alice"],
             ["David", "Daniel", "Adam", "Leo", "Max", "Lucas", "Marco", "Peter", "Paul", "Simon", "Alex",
              "Thomas", "Michael", "Oliver", "James", "Tom"]),
}
ORIGIN_KEYWORDS = [  # checked in order against the voice description (informational only)
    ("british", "gb"), ("american", "us"), ("australian", "au"), ("canadian", "ca"),
]


# ----------------------------------------------------------------------------- sampling
def wchoice(rng: random.Random, weights: dict):
    keys = [k for k, w in weights.items() if w > 0]
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def load_voices(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, list):  # spec file
        data = {v["voice_id"]: v for v in data}
    for vid, v in data.items():
        desc = v.get("description", "").lower()
        v.setdefault("gender", "female" if desc.startswith(("female", "woman")) else "male")
        v.setdefault("origin", next((o for k, o in ORIGIN_KEYWORDS if k in desc), "intl"))
        v.setdefault("tier", "pool")
    return data


def outcome_weights(perf: str, unfixable: bool, group: str) -> dict:
    w = dict(tax.OUTCOME_BY_PERFORMANCE[perf])
    if unfixable:
        moved = w["resolved"] * 0.6
        w["resolved"] -= moved
        w["partial"] += moved
    if group in ("neutral", "positive"):  # nothing to escalate to the purser
        w["partial"] += w.pop("escalated")
    return w


def sample_seed(i: int, seed: int, voices: dict) -> dict:
    rng = random.Random(f"{seed}-{i}")
    group = wchoice(rng, tax.GROUP_WEIGHTS)
    cats = [c for c, d in tax.CATEGORIES.items() if d["group"] == group]
    cat = rng.choice(cats)
    spec = tax.CATEGORIES[cat]

    phase = rng.choice(spec["phases"])
    if phase == "cruise_night":
        flight = "long_haul_night"
    elif spec.get("long_only"):
        flight = rng.choice(["long_haul_day", "long_haul_night"])
    else:
        flight = rng.choice(list(tax.FLIGHT_TYPES))
    cabin = wchoice(rng, tax.CABINS_LONG if flight.startswith("long") else tax.CABINS_SHORT)

    emo, inten = rng.choice(spec["baseline"])
    perf = wchoice(rng, tax.PERFORMANCE_BY_GROUP[group])
    mistake = ""
    if perf in ("poor", "adequate"):
        mistake = rng.choice(spec.get("mistakes", []) + tax.GENERIC_MISTAKES)
    outcome = wchoice(rng, outcome_weights(perf, spec.get("unfixable", False), group))
    kind = group if group in ("conflict", "distress") else "calm"
    arc = rng.choice(tax.ARCS[(kind, outcome)])

    pax = [v for v, d in voices.items() if d["role"] == "passenger"]
    heroes = [v for v in pax if voices[v]["tier"] == "hero"]
    if group in ("conflict", "distress") and heroes:
        pax_id = rng.choice(heroes)
    else:
        pax_id = rng.choice(pax)
    crew_id = rng.choice([v for v, d in voices.items() if d["role"] == "crew"])
    pv = voices[pax_id]
    female, male = NAME_POOLS["intl"]
    name = rng.choice(female if pv["gender"] == "female" else male)

    return {
        "index": i, "group": group, "category": cat, "phase": phase, "flight_type": flight,
        "cabin_class": cabin, "goal": rng.choice(spec["situations"]),
        "baseline_emotion": emo, "baseline_intensity": inten,
        "performance": perf, "mistake": mistake, "outcome": outcome, "arc": arc,
        "profanity": wchoice(rng, tax.PROFANITY_BY_GROUP[group]),
        "opener": "crew" if rng.random() < tax.CREW_OPENS[group] else "passenger",
        "modifier": rng.choice(tax.MODIFIERS) if rng.random() < 0.5 else "",
        "passenger_voice": pax_id, "crew_voice": crew_id, "passenger_name": name,
    }


# ----------------------------------------------------------------------------- prompt
def load_prompt(path: Path) -> tuple[str, str]:
    """File = notes --- instructions --- scenario block. Returns (system, user_template)."""
    parts = re.split(r"\n---\n", path.read_text(encoding="utf-8"))
    if len(parts) != 3:
        raise SystemExit(f"{path}: expected 'notes --- instructions --- scenario' (got {len(parts)} parts)")
    system, user = parts[1].strip(), parts[2].strip()
    if "{{" in system:
        raise SystemExit("placeholders in the instruction part break prompt caching; keep them in the scenario block")
    return system, user


def fill(template: str, seed: dict, voices: dict) -> str:
    pv = voices[seed["passenger_voice"]]
    fields = {
        **seed,
        "phase": tax.PHASES[seed["phase"]],
        "flight_type": tax.FLIGHT_TYPES[seed["flight_type"]],
        "cabin_class": seed["cabin_class"].replace("_", " "),
        "baseline_emotion": f"{seed['baseline_emotion']}, intensity {seed['baseline_intensity']}",
        "mistake_if_any": f"Mistake: {seed['mistake']}." if seed["mistake"] else "",
        "passenger_profile": pv["description"],
        "modifier": seed["modifier"] or "nothing special",
    }

    def sub(m: re.Match) -> str:
        if m.group(1) not in fields:
            raise KeyError(f"no value for placeholder {{{{{m.group(1)}}}}}")
        return str(fields[m.group(1)])

    return re.sub(r"\{\{(\w+)\}\}", sub, template)


# ----------------------------------------------------------------------------- checks
def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def fix_overlaps(turns: list[dict]) -> list[str]:
    """Repair overlap markup in place. Returns notes. Backchannels need a long floor turn, an
    'at' phrase that exists in it, and a different speaker; interruptions need a different speaker
    and the interrupted turn pre-cut with '...'."""
    notes, floor, out = [], None, []
    for t in turns:
        same = floor is not None and t.get("speaker") == floor.get("speaker")
        if t.get("backchannel"):
            ft = (floor or {}).get("text", "")
            if floor is None or same or len(words(ft)) < 8:
                notes.append("dropped backchannel (short turn or same speaker)")
                continue
            at = t.get("at")
            if at and " ".join(words(at)) not in " ".join(words(ft)):
                t.pop("at")
                notes.append("removed 'at' not found in floor turn")
        elif t.get("interrupts"):
            ft = (floor or {}).get("text", "").rstrip()
            if same or floor is None or not ft.endswith(("...", "\u2026", "-")) or len(words(ft)) < 6:
                t.pop("interrupts")
                t["gap_s"] = 0.1
                notes.append("interrupt not possible -> quick reply")
        out.append(t)
        if not t.get("backchannel"):
            floor = t
    turns[:] = out
    return notes


EMOTION_SYNONYMS = {
    "frustrated": "irritated", "annoyed": "irritated", "impatient": "irritated", "upset": "distressed",
    "furious": "angry", "mad": "angry", "livid": "angry", "scared": "anxious", "nervous": "anxious",
    "worried": "anxious", "afraid": "anxious", "tearful": "sad", "grieving": "sad", "ashamed": "embarrassed",
    "sheepish": "embarrassed", "thankful": "grateful", "calm": "neutral",
}


def repair(turns: list[dict], seed: dict) -> list[str]:
    """Fix what is mechanically fixable instead of spending a whole retry (~2 min each) on it.
    Returns notes. Anything not fixable here is still rejected by validate_turns / lint."""
    notes: list[str] = []
    prev = (seed["baseline_emotion"], seed["baseline_intensity"])
    for i, t in enumerate(turns):
        if isinstance(t.get("text"), str) and any(c.isdigit() for c in t["text"]):
            t["text"] = numbers_to_words(t["text"])
            notes.append("digits->words")
        if t.get("speaker") != "passenger":
            continue
        emo = t.get("emotion")
        if isinstance(emo, str):
            emo = emo.strip().lower()
            emo = EMOTION_SYNONYMS.get(emo, emo)
            if emo != t.get("emotion"):
                notes.append("emotion label normalised")
        inten = t.get("intensity")
        if isinstance(inten, (float, str)):
            try:
                inten = int(round(float(inten)))
            except ValueError:
                inten = None
        if emo not in VALID_EMOTIONS:
            if emo is None:  # missing: keep the previous passenger state
                emo, inten = prev[0], prev[1] if inten is None else inten
                notes.append("missing emotion inherited")
        if inten is None:
            inten = prev[1]
            notes.append("missing intensity inherited")
        inten = max(0, min(3, inten))
        if emo in VALID_EMOTIONS and prev[1] - inten > 1 and not t.get("backchannel"):
            inten = prev[1] - 1  # calming is one step per turn
            notes.append("calm-down clamped to one step")
        t["emotion"], t["intensity"] = emo, inten
        if not t.get("backchannel"):
            prev = (emo, inten)
    return notes


def lint(script: dict, seed: dict) -> tuple[list[str], list[str]]:
    """(hard, soft). Hard problems cost a retry. Soft ones are recorded in gen.soft and accepted:
    a missing 'trigger' label is not worth two minutes of GPU (triggers are not used in training)."""
    hard, soft, turns = [], [], script["turns"]
    if not 10 <= len(turns) <= 34:
        hard.append(f"{len(turns)} turns")
    if turns and turns[0].get("speaker") != seed["opener"]:
        hard.append(f"opener is {turns[0].get('speaker')}, wanted {seed['opener']}")
    bw = len(words(script["passenger"]["brief"]))
    if not 45 <= bw <= 150:
        hard.append(f"brief has {bw} words")
    prev = None
    for i, t in enumerate(turns):
        if t.get("speaker") != "passenger":
            continue
        low = t.get("text", "").lower()
        if len(words(low)) > MAX_PASSENGER_WORDS:
            hard.append(f"turn {i}: passenger turn over {MAX_PASSENGER_WORDS} words")
        hit = next((b for b in BANNED_PASSENGER if b in low), None)
        if hit:
            hard.append(f"turn {i}: banned phrase {hit!r}")
        cur = (t.get("emotion"), t.get("intensity"))
        if prev is not None and cur != prev and not str(t.get("trigger") or "").strip():
            soft.append(f"turn {i}: emotion changed without trigger")
        prev = cur
    last = next((t for t in reversed(turns) if t.get("speaker") == "passenger" and not t.get("backchannel")), None)
    if last:
        emo, inten = last.get("emotion"), last.get("intensity", 0)
        if seed["outcome"] == "resolved" and emo not in CALM_EMOTIONS and inten > 1:
            hard.append(f"outcome resolved but passenger ends {emo} {inten}")
        if seed["outcome"] == "unresolved" and emo in ("relieved", "grateful"):
            hard.append(f"outcome unresolved but passenger ends {emo}")
    if seed["outcome"] == "escalated" and not any(
            re.search(r"\b(purser|captain)\b", t.get("text", "").lower()) for t in turns if t.get("speaker") == "crew"):
        hard.append("outcome escalated but crew never mentions the purser or captain")
    return hard, soft


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in reply")
    return json.loads(text[a:b + 1])


# ----------------------------------------------------------------------------- generation
def make_script(seed: dict, reply: dict) -> dict:
    turns = reply.get("turns")
    if not isinstance(turns, list):
        raise ValueError("reply has no 'turns' list")
    for t in turns:  # normalise light model mistakes
        if isinstance(t.get("intensity"), str) and t["intensity"].isdigit():
            t["intensity"] = int(t["intensity"])
        if t.get("speaker") == "crew":
            for k in ("emotion", "intensity", "trigger"):
                t.pop(k, None)
    scen = {k: seed[k] for k in ("group", "category", "phase", "flight_type", "cabin_class", "goal",
                                 "outcome", "arc", "profanity", "opener", "modifier")}
    return {
        "id": f"cc_{seed['index']:06d}",
        "scenario": scen,
        "passenger": {"voice_id": seed["passenger_voice"], "name": seed["passenger_name"],
                      "baseline_emotion": seed["baseline_emotion"],
                      "brief": str(reply.get("brief", "")).strip()},
        "crew": {"voice_id": seed["crew_voice"], "performance": seed["performance"], "mistake": seed["mistake"]},
        "outcome": seed["outcome"],
        "turns": turns,
    }


class Meter:
    """Thread-safe counters so the run can report real tokens/s and the retry rate."""

    def __init__(self):
        self.lock = threading.Lock()
        self.attempts = self.tokens = self.thinking = 0
        self.reasons: Counter = Counter()

    def add(self, tokens: int = 0, thinking: bool = False):
        with self.lock:
            self.attempts += 1
            self.tokens += tokens
            self.thinking += bool(thinking)

    def reject(self, reason: str):
        with self.lock:
            self.reasons[re.sub(r"turn \d+: ", "", reason)[:60]] += 1


METER = Meter()


def log_attempt(args, seed: dict, attempt: int, reasons: list[str]) -> None:
    """Every rejected attempt, also for scripts that later succeed: shows what to fix next."""
    for r in reasons:
        METER.reject(r)
    with open(Path(args.out_dir) / "_attempts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"index": seed["index"], "attempt": attempt, "reasons": reasons}) + "\n")


def generate_one(seed: dict, client, args, system: str, template: str, voices: dict) -> tuple[str, dict | None]:
    out_path = Path(args.out_dir) / f"cc_{seed['index']:06d}.json"
    if out_path.exists():
        return "skipped", None
    user = fill(template, seed, voices)
    reasons: list[str] = []
    for attempt in range(args.max_attempts):
        try:
            kw = dict(model=args.model, temperature=args.temperature + 0.05 * attempt, top_p=0.95,
                      max_tokens=args.max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            if not args.no_json_mode:
                kw["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kw)
            msg = resp.choices[0].message
            extra = getattr(msg, "model_extra", None) or {}
            METER.add(getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0,
                      thinking=bool(extra.get("reasoning_content")) or "<think>" in (msg.content or ""))
            script = make_script(seed, parse_json(msg.content or ""))
        except Exception as e:  # network, JSON, schema
            reasons.append(f"{type(e).__name__}: {str(e)[:120]}")
            log_attempt(args, seed, attempt + 1, [reasons[-1]])
            continue
        notes = repair(script["turns"], seed) + fix_overlaps(script["turns"])
        hard, soft = lint(script, seed)
        errs = validate_turns(script["turns"]) + hard
        if not errs:
            script["gen"] = {"model": args.model, "attempt": attempt + 1, "fixes": sorted(set(notes)), "soft": soft}
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(script, indent=1, ensure_ascii=False))
            tmp.replace(out_path)
            return f"ok (attempt {attempt + 1}{', ' + str(len(set(notes))) + ' fixes' if notes else ''})", script
        reasons.append("; ".join(errs[:4]))
        log_attempt(args, seed, attempt + 1, errs)
    with open(Path(args.out_dir) / "_failed.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"seed": seed, "reasons": reasons}, ensure_ascii=False) + "\n")
    return "FAILED: " + (reasons[-1] if reasons else "?"), None


def summarize(out_dir: Path) -> dict:
    c = {k: Counter() for k in ("group", "category", "outcome", "profanity", "performance", "passenger_voice")}
    emo, fixes = Counter(), Counter()
    n = turns = pax_words = pax_turns = 0
    for p in out_dir.glob("cc_*.json"):
        s = json.loads(p.read_text())
        n += 1
        for k in ("group", "category", "outcome", "profanity"):
            c[k][s["scenario"][k]] += 1
        c["performance"][s["crew"]["performance"]] += 1
        c["passenger_voice"][s["passenger"]["voice_id"]] += 1
        turns += len(s["turns"])
        for t in s["turns"]:
            if t["speaker"] == "passenger":
                pax_turns += 1
                pax_words += len(t["text"].split())
                if not t.get("backchannel"):
                    emo[f"{t['emotion']}_{t['intensity']}"] += 1
        fixes.update(s.get("gen", {}).get("fixes", []))
    stats = {"scripts": n, "turns_mean": round(turns / max(n, 1), 1),
             "passenger_words_per_turn": round(pax_words / max(pax_turns, 1), 1),
             **{k: dict(v.most_common()) for k, v in c.items()},
             "passenger_emotion_intensity": dict(emo.most_common()), "auto_fixes": dict(fixes.most_common())}
    (out_dir / "_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True, help="number of dialogue indices to generate")
    p.add_argument("--start", type=int, default=0, help="first index (ids are cc_<index>)")
    p.add_argument("--voices", required=True, help="voice_specs.json or voices.json registry")
    p.add_argument("--prompt", default="prompts/script_generation_prompt.md")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="local")
    p.add_argument("--workers", type=int, default=16, help="match llama-server --parallel")
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--max_tokens", type=int, default=3000)
    p.add_argument("--max_attempts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_json_mode", action="store_true", help="if the server rejects response_format")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    voices = load_voices(Path(args.voices))
    if not any(v["role"] == "crew" for v in voices.values()) or not any(v["role"] == "passenger" for v in voices.values()):
        raise SystemExit("need at least one passenger and one crew voice")
    system, template = load_prompt(Path(args.prompt))
    seeds = [sample_seed(i, args.seed, voices) for i in range(args.start, args.start + args.n)]

    if args.dry_run:
        for k in ("group", "category", "outcome", "performance", "profanity", "opener"):
            print(k, dict(Counter(s[k] for s in seeds).most_common()))
        print("\n----- system prompt: %d chars\n----- example user prompt:\n%s" % (len(system), fill(template, seeds[0], voices)))
        return

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=600)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(generate_one, s, client, args, system, template, voices): s for s in seeds}
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                status, _ = fut.result()
            except Exception as e:
                status = f"ERROR {type(e).__name__}: {e}"
            done += status.startswith("ok")
            failed += status.startswith(("FAILED", "ERROR"))
            print(f"[{k}/{len(seeds)}] cc_{s['index']:06d} {s['group']}/{s['category']}: {status} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    stats = summarize(out_dir)
    el = max(time.time() - t0, 1e-9)
    print(f"written {done}, failed {failed}; {stats['scripts']} scripts in {out_dir / '_stats.json'}", file=sys.stderr)
    print(f"{METER.attempts} LLM calls for {len(seeds)} scripts ({METER.attempts / max(len(seeds), 1):.2f} per script), "
          f"{METER.tokens} output tokens = {METER.tokens / el:.0f} tok/s aggregate, {el / max(done, 1):.0f} s per script",
          file=sys.stderr)
    if METER.thinking:
        print(f"WARNING: {METER.thinking} replies contained thinking text; the server ignored enable_thinking=false "
              "(start llama-server with --reasoning-budget 0 or --chat-template-kwargs '{\"enable_thinking\":false}')",
              file=sys.stderr)
    print("most common rejection reasons:", METER.reasons.most_common(6), file=sys.stderr)


if __name__ == "__main__":
    main()