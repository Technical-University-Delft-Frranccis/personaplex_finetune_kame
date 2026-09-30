#!/usr/bin/env bash
# PersonaPlex + KAME oracle fine-tune: data generation -> training -> export.
# Two environments: .venv-tts (qwen-tts, Python 3.12) for steps 0-1, the kame-finetune venv for the rest.
# Every step is resumable (finished files are skipped).
set -euo pipefail

DATA=data/cabin                        # put scripts/ and voice_specs.json here
PROC=processed_data/cabin
VOICES=$DATA/voices.json               # written by step 0, read by steps 1-2
LLM_URL=http://localhost:8000/v1       # vLLM serving the RUNTIME LLM
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct     # must equal server_oracle --llm-model
FT_INIT=/workspace/ft_init             # from tools/init_for_ft.py (already done)

# --- 0. once: loss mask over the system prompt (idempotent)
#python patches/apply_prompt_mask_patch.py utils/data.py

# --- 0b. voice bank (TTS venv): design voices, fix anchors, emotion references
#   source .venv-tts/bin/activate
#python -m tools.build_voice_bank --spec $DATA/voice_specs.json --out_dir $DATA/voices --registry $VOICES
#   -> LISTEN to $DATA/voices/*/ ; delete an entry from voices.json to rebuild that voice

# --- 1. render turns (TTS venv), one process per GPU
.venv-tts/bin/python -m tools.synthesize_turns --scripts_dir $DATA/scripts --voices $VOICES \
  --out_dir $DATA/turns --device cuda:0 --shard 0 --num_shards 1 --min_similarity 0
#   deactivate 

source .venv/bin/activate
# --- 2. align + assemble (kame-finetune venv; uv pip install pyloudnorm)
uv run --no-sync -m tools.assemble_dialogues --turns_dir $DATA/turns --voices $VOICES --out_dir $DATA

exit 0

# --- 3. oracles with the runtime LLM and the shared persona template
uv run --extra oracle -m tools.generate_oracle_local \
    --text_dir $DATA/text --meta_dir $DATA/meta --output_dir $DATA/oracle_raw \
    --base_url $LLM_URL --model $LLM_MODEL --system_prompt prompts/oracle_system_prompt.txt --workers 16

# --- 4. KAME tokenizers (PersonaPlex repo ships the same Mimi + SentencePiece files)
uv run -m tools.tokenize_audio --audio_dir $DATA/audio --output_dir $DATA/tokenized_audio \
    --audio_tokenizer_repo nvidia/personaplex-7b-v1
uv run -m tools.tokenize_text --word_transcript_dir $DATA/text --output_dir $DATA/tokenized_text \
    --text_tokenizer_repo nvidia/personaplex-7b-v1
uv run -m tools.tokenize_oracle --oracle_dir $DATA/oracle_raw --oracle_suffix ".json" \
    --tokenized_audio_dir $DATA/tokenized_audio --output_dir $DATA/tokenized_oracle \
    --A_channel 0 --B_channel 1
uv run -m tools.prepare_dataset --tokenized_text_dir $DATA/tokenized_text \
    --tokenized_audio_dir $DATA/tokenized_audio --tokenized_oracle_dir $DATA/tokenized_oracle \
    --output_prefix $PROC/raw

# --- 5. PersonaPlex hybrid prompt (voice prompt = the voice's prompt.wav, full clip)
uv run -m tools.add_personaplex_prompt --parquet_glob "$PROC/raw-*.parquet" \
    --meta_dir $DATA/meta --output_prefix $PROC/train --max_frames 2400

# --- 6. train: context control, depth transformer frozen
uv run accelerate launch --num_processes 1 --num_machines 1 --use_deepspeed \
    --deepspeed_config_file ds_configs/zero3-bfp16-warmlr-act_ckpt.json \
    finetune.py --launcher accelerate \
    --model_dir $FT_INIT --model_dtype bfloat16 --output_dir output/pp-oracle-pilot \
    --train_data_files "$PROC/train-*.parquet" \
    --moshi_speakers A --model_user_stream --use_oracle \
    --max_length 2402 --min_length 256 --parameters_to_finetune tempformer \
    --tempformer_learning_rate 4e-6 --depformer_learning_rate 4e-6 \
    --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --save_steps 200

# --- 7. export (do NOT run clean_moshi: it strips dep_q back to 8)
#   uv run -m tools.zero_to_fp32 output/pp-oracle-pilot/step_XXX output/pp-oracle-pilot/step_XXX_fp32 \
#       --moshi_lm_kwargs_path $FT_INIT/moshi_lm_kwargs.json
#   uv run -m tools.convert_to_pp --ft_dir output/pp-oracle-pilot/step_XXX_fp32 \
#       --out /workspace/checkpoints/personaplex_oracle_ft.safetensors
# Then, in personaplex-klm: PERSONAPLEX_ORACLE_CHECKPOINT=... pytest tests/test_checkpoint_load.py tests/test_oracle.py
# and regenerate every custom_voices/*.pt from its prompt.wav with the fine-tuned model.