#!/bin/bash
# Mode B2: Train everything in DiT EXCEPT cross_attn (cross_attn frozen).
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}

DATA_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
OUT="/home/zeqingwang/zeqingwang/models/final_model/sf2/baseline/scoped_no_cross_attn"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

echo "=== SF2_final / Mode B2: train all DiT except cross_attn ==="
echo "Output: ${OUT}"
echo "Start:  $(date)"

PYTHONUNBUFFERED=1 accelerate launch --num_processes=3 --multi_gpu --main_process_port 29603 \
  examples/ReactiveGWM/model_training/train.py \
  --game sf2 \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${DATA_ROOT}/metadata.csv" \
  --data_file_keys "video,action" \
  --height 480 --width 608 --num_frames 101 \
  --dataset_repeat 1 \
  --model_paths "[[\"${BASE}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]" \
  --tokenizer_path "/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --learning_rate 5e-5 \
  --num_epochs 1000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 2 \
  --output_path "${OUT}" \
  --use_gradient_checkpointing \
  --trainable_models "dit" \
  --trainable_filter_exclude "cross_attn" \
  --extra_inputs "input_image" \
  --action_hold_window 10 \
  --use_csv_prompt true \
  --prompt_column prompt \
  --dataset_num_workers 4

echo "Finished at $(date)"
