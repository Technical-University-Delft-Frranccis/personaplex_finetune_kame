#!/usr/bin/env bash
# Overnight chain: voice bank -> TTS turns -> voice-consistency filter -> assemble dialogues.
# Stops BEFORE the oracle pass, tokenising and fine-tuning (those need the other LLM / a decision).
#
# Run from the repo root:
#   mkdir -p logs
#   TTS_PY=/path/to/tts-env/bin/python nohup bash run_overnight.sh > logs/overnight.log 2>&1 &
#
# Every stage is resumable (voice bank skips voices already in the registry, synthesize_turns skips
# finished dialogues, assemble skips finished ids), so if the pod dies just start it again.
set -u
cd "$(dirname "$0")"

: "${TTS_PY:?set TTS_PY to the python executable of the Qwen3-TTS environment}"
NUM_SHARDS=${NUM_SHARDS:-3}     # parallel TTS processes on the one GPU (watch nvidia-smi; lower if OOM)
KAME_RUN=${KAME_RUN:-uv run}    # how to run things in the kame_finetune environment
DATA=${DATA:-data/cabin}
mkdir -p logs

stamp() { echo "[$(date '+%F %T')] $*"; }
die() { stamp "ABORT: $*"; exit 1; }

# ---------------------------------------------------------------- pre-flight
[ -x "$TTS_PY" ] || die "TTS_PY=$TTS_PY is not executable"
N_SCRIPTS=$(ls "$DATA"/scripts/cc_*.json 2>/dev/null | wc -l)
[ "$N_SCRIPTS" -gt 0 ] || die "no scripts in $DATA/scripts"
# if grep -q "paid for this seat" tools/build_voice_bank.py; then
#   die "EMOTION_LEVELS patch not applied (patches/voice_bank_emotion_levels.py -> tools/build_voice_bank.py)"
# fi
# grep -q "_at_time" tools/assemble_dialogues.py \
#   || die "place_turns patch not applied (patches/assemble_place_turns.py -> tools/assemble_dialogues.py)"
N_SPEC=$("$TTS_PY" -c "import json;print(len(json.load(open('voice_specs.json'))))") || die "cannot read voice_specs.json"
stamp "pre-flight ok: $N_SCRIPTS scripts, $N_SPEC voices, $NUM_SHARDS TTS shards"

# ---------------------------------------------------------------- 0. free the GPU
pkill -f llama-server && stamp "stopped llama-server" && sleep 8
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# ---------------------------------------------------------------- 1. voice bank
stamp "1/4 voice bank"
"$TTS_PY" -m tools.build_voice_bank --spec voice_specs.json --out_dir "$DATA/voices" \
    --registry "$DATA/voices.json" > logs/voice_bank.log 2>&1
N_REG=$("$TTS_PY" -c "import json;print(len(json.load(open('$DATA/voices.json'))))" 2>/dev/null || echo 0)
[ "$N_REG" -ge "$N_SPEC" ] || die "voice bank incomplete ($N_REG/$N_SPEC voices), see logs/voice_bank.log"
stamp "voice bank done: $N_REG voices"

# ---------------------------------------------------------------- 2. TTS
stamp "2/4 synthesize turns ($NUM_SHARDS shards)"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  "$TTS_PY" -m tools.synthesize_turns --scripts_dir "$DATA/scripts" --voices "$DATA/voices.json" \
      --out_dir "$DATA/turns" --device cuda:0 --shard "$i" --num_shards "$NUM_SHARDS" \
      > "logs/tts_$i.log" 2>&1 &
done
wait
N_TURNS=$(ls "$DATA"/turns/*/manifest.json 2>/dev/null | wc -l)
stamp "TTS done: $N_TURNS/$N_SCRIPTS dialogues rendered"
[ "$N_TURNS" -gt 0 ] || die "no dialogues rendered, see logs/tts_*.log"

# ---------------------------------------------------------------- 3. voice-consistency filter
stamp "3/4 voice consistency filter"
"$TTS_PY" -m tools.filter_voice_consistency --turns_dir "$DATA/turns" --quarantine > logs/filter.log 2>&1
tail -n 1 logs/filter.log

# ---------------------------------------------------------------- 4. assemble
stamp "4/4 assemble dialogues"
$KAME_RUN -m tools.assemble_dialogues --turns_dir "$DATA/turns" --voices "$DATA/voices.json" \
    --out_dir "$DATA" > logs/assemble.log 2>&1
tail -n 2 logs/assemble.log

N_OUT=$(ls "$DATA"/meta/*.json 2>/dev/null | wc -l)
stamp "FINISHED: $N_OUT assembled dialogues in $DATA/{audio,text,meta}. Not started: oracle pass, tokenising, fine-tune."
