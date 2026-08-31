#!/bin/bash
# Baseline ablation: scoped full DiT (cross_attn frozen) + fixed English prompt
# "Street Fighter 2 arcade fighting game gameplay" + cold start.
#
# Derived from full_5s_480x608_cached_v5_p4.sh. Differences:
#   * --trainable_filter_exclude "cross_attn"     (freeze cross_attn block)
#   * --use_csv_prompt false                      (route through SF2_FIXED_PROMPT)
#   * Removed --resume_from_ckpt                  (cold start)
#   * CACHE_ROOT -> clips_5s_cache_v5_fixedprompt (rebuilt with fixed-prompt T5)
#   * OUT       -> .../Baseline/no_cross_fixed_prompt
#   * main_process_port -> 29608                  (avoid clash with p4 / 29606)
#
# Cache contract (must hold before launch):
#   examples/ReactiveGWM/scripts/rebuild_cache_fixed_prompt.py has produced
#   ${CACHE_ROOT}/manifest.json with config.use_csv_prompt == False and
#   every row's prompt_hash pointing to T5("Street Fighter 2 arcade fighting
#   game gameplay"). CachedSF2Dataset._assert_config_match validates this at
#   __init__ — mismatch -> immediate RuntimeError.
#
# Smoke checks (look for these in the first ~30s of stdout):
#   [Cold-start] Transferred N/M keys                 (and NO [Resume] line)
#   [Trainable Filter] keep=[] drop=['cross_attn'] -> X tensors trainable (Y scalars)
#   [cached] filtered 4 pipeline units
#   [cached] dropped pipe.vae and pipe.text_encoder

set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

DATA_ROOT="${DATA_ROOT:-/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s}"
CACHE_ROOT="${CACHE_ROOT:-/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s_cache_v5_fixedprompt}"
METADATA="${METADATA:-${DATA_ROOT}/metadata_v5.csv}"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
OUT="${OUT:-/home/zeqingwang/zeqingwang/models/train/sf2/Baseline/no_cross_fixed_prompt}"

DATASET_REPEAT="${DATASET_REPEAT:-1}"

if [ ! -f "${CACHE_ROOT}/manifest.json" ]; then
  echo "ERROR: cache manifest missing at ${CACHE_ROOT}/manifest.json"
  echo "       Run examples/ReactiveGWM/scripts/rebuild_cache_fixed_prompt.py first."
  exit 1
fi

mkdir -p "${OUT}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

echo "=== SF2_final / Baseline: scoped full DiT (NO cross_attn) + fixed prompt ==="
echo "Data:    ${DATA_ROOT}"
echo "CSV:     ${METADATA}"
echo "Cache:   ${CACHE_ROOT}"
echo "Output:  ${OUT}"
echo "Prompt:  Street Fighter 2 arcade fighting game gameplay  (via SF2_FIXED_PROMPT)"
echo "GPUs:    ${CUDA_VISIBLE_DEVICES}"
echo "Start:   $(date)"

PYTHONUNBUFFERED=1 accelerate launch --num_processes=8 --multi_gpu --main_process_port 29608 \
  examples/ReactiveGWM/model_training/train.py \
  --game sf2 \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${METADATA}" \
  --data_file_keys "video,action" \
  --height 480 --width 608 --num_frames 101 \
  --dataset_repeat "${DATASET_REPEAT}" \
  --use_cached_dataset --cache_root "${CACHE_ROOT}" \
  --model_paths "[[\"${BASE}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]" \
  --tokenizer_path "/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --learning_rate 5e-5 \
  --num_epochs 1000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 1 \
  --output_path "${OUT}" \
  --use_gradient_checkpointing \
  --trainable_models "dit" \
  --trainable_filter_exclude "cross_attn" \
  --extra_inputs "input_image" \
  --action_hold_window 10 \
  --use_csv_prompt false \
  --dataset_num_workers 4 \
  --prompt_dropout_prob 0.1

echo "Finished at $(date)"
