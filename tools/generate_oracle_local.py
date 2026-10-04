"""Step 3 - generate oracle_raw/*.json with the LOCAL runtime LLM (vLLM / llama.cpp / LM Studio).

Reuses KAME's OracleGenerator (event timing, spoken-ratio bookkeeping, hint = ground-truth
next passenger utterance). What is different from KAME's OpenAI script is the PROMPT: it is
built in the same format server_oracle.py uses live, so training oracles look like runtime
oracles:
  system  = prompts/oracle_system_prompt.txt with {{BRIEF}} replaced by the passenger brief
            (the same file server_oracle.py loads with --persona-prompt-file)
  user    = the conversation so far with "passenger:" / "crew:" labels, including the crew
            member's partial turn (crew text normalised like Vosk output: lower case, no punctuation)
            + (only when > 50% of the crew turn is heard) a hidden-hint block, following
              KAME's graded hint levels
            + ORACLE_TAIL (tools/oracle_text.py, identical copy in the server)
Only the passenger's channel (A = 0) gets oracles: the model never plays the crew.

Differences from the first version (pilot findings: ~20% of predictions were written in the CREW's voice,
and predictions did not converge to the hint at later levels):
  * ORACLE_TAIL stops the model completing the crew's unfinished sentence
  * ratio >= 0.95: the hint IS the oracle (KAME levels 4-5; no LLM call)
  * ratio 0.8-0.95: the instruction asks for the hint's meaning in slightly different words
  * a crew-sounding prediction is re-sampled up to --crew_retries times at lower temperature; if it still
    sounds like crew it is dropped (first half) or replaced by the hint (second half)

    uv run --extra oracle -m tools.generate_oracle_local \
        --text_dir data/cabin/text --meta_dir data/cabin/meta --output_dir data/cabin/oracle_raw \
        --base_url http://localhost:8000/v1 --model llama32-3b-unc --workers 16
Use the SAME --model as the live server's --llm-model.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from tools.generate_oracle_from_text import prediction_to_record
from tools.oracle_generation import OracleGenerator, OraclePredictionRequest, words_from_word_transcript
from tools.oracle_text import build_user_message, clean_full, looks_like_crew, normalize_crew_text

PASSENGER_CHANNEL = 0
SPEAKER_TO_CHANNEL = {"A": 0, "B": 1}
LABELS = {"A": "passenger", "B": "crew"}  # must match MODEL_SPEAKER / USER_SPEAKER in server_oracle.py
HINT_VERBATIM_FROM = 0.95                 # KAME levels 4-5: the oracle is the hint itself


def relabel_context(context: str) -> str:
    """KAME writes 'A: ...' / 'B: ...'; the live server writes 'passenger: ...' / 'crew: ...'.
    Crew lines are normalised like the live ASR (Vosk): lower case, no punctuation."""
    out = []
    for line in context.splitlines():
        head, sep, rest = line.partition(":")
        if not sep:
            out.append(line)
            continue
        label = LABELS.get(head.strip(), head.strip())
        rest = " " + normalize_crew_text(rest) if label == "crew" else rest
        out.append(f"{label}:{rest}")
    return "\n".join(out)


def hint_block(req: OraclePredictionRequest) -> str:
    """KAME's hint levels, compressed. Empty in the first half of the crew turn."""
    r = req.current_spoken_ratio
    if r <= 0.5:
        return ""
    if r <= 0.65:
        how = "Rely mainly on the conversation so far; borrow at most a few keywords from the hint."
    elif r <= 0.8:
        how = "Follow the conversation and use the hint, but include some content that differs from it."
    else:
        how = "Say what the hint says, in slightly different words."
    return ("\n\n(Hidden hint, never mention it: the passenger's next words will be similar to: "
            f"\"{req.next_utterance_hint}\". {how})")


def build_messages(req: OraclePredictionRequest, system_prompt: str) -> list[dict]:
    user = build_user_message(relabel_context(req.conversation_context), hint_block(req))
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def make_predict_fn(client: OpenAI, model: str, system_prompt: str, temperature: float, crew_retries: int = 2):
    def predict(req: OraclePredictionRequest) -> str:
        hint = req.next_utterance_hint
        if req.current_spoken_ratio >= HINT_VERBATIM_FROM and hint:
            return hint
        fallback = hint if req.current_spoken_ratio > 0.5 else ""
        messages = build_messages(req, system_prompt)
        for attempt in range(crew_retries + 1):
            temp = temperature if attempt == 0 else max(0.3, temperature - 0.3)
            try:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=temp, max_tokens=80)
                out = clean_full(resp.choices[0].message.content or "")
            except Exception:
                return fallback
            if len(out.split()) >= 2 and not looks_like_crew(out):
                return out
        return fallback  # still unusable: drop in the first half, use the hint in the second

    return predict


def process(text_path: Path, meta_dir: Path, out_dir: Path, client: OpenAI, args, template: str) -> str:
    out_path = out_dir / text_path.name
    if out_path.exists():
        return "skipped (done)"
    meta = json.loads((meta_dir / text_path.name).read_text())
    system_prompt = template.replace("{{BRIEF}}", meta["prompt_text"].strip())
    words = words_from_word_transcript(json.loads(text_path.read_text()))
    generator = OracleGenerator(
        make_predict_fn(client, args.model, system_prompt, args.temperature, args.crew_retries),
        time_interval=args.time_interval, target_channel=PASSENGER_CHANNEL,
        speaker_to_channel=SPEAKER_TO_CHANNEL,
    )
    records = [prediction_to_record(p) for p in generator.generate_predictions(words)]
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=1))
    tmp.replace(out_path)
    return f"ok {len(records)} events"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text_dir", required=True)
    p.add_argument("--meta_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--system_prompt", default="prompts/oracle_system_prompt.txt")
    p.add_argument("--time_interval", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--crew_retries", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    template = Path(args.system_prompt).read_text(encoding="utf-8")
    if "{{BRIEF}}" not in template:
        raise SystemExit(f"{args.system_prompt} must contain {{{{BRIEF}}}}")
    client = OpenAI(base_url=args.base_url, api_key="local")
    served = [m.id for m in client.models.list().data]
    if args.model not in served:
        raise SystemExit(f"LLM server serves {served}, not {args.model!r}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = sorted(Path(args.text_dir).glob("*.json"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, t, Path(args.meta_dir), out_dir, client, args, template): t for t in texts}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                status = fut.result()
            except Exception as e:
                status = f"ERROR {type(e).__name__}: {e}"
            print(f"[{k}/{len(texts)}] {futures[fut].stem}: {status}", flush=True)


if __name__ == "__main__":
    main()