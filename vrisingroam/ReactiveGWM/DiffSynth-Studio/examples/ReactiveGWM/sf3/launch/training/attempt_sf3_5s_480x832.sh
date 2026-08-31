#!/bin/bash
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "=== SF3 Phase 1 at 480x832, 5s, NON-CACHED baseline ==="
echo ""
echo "Motivation: reference baseline for cached variant's loss parity gate."
echo "  Runs VAE + T5 on-the-fly per step (~2.96 s/step expected at 480x832)."
echo "  For production training prefer attempt_sf3_5s_480x832_cached.sh."
echo ""
echo "Resolution design (SF3 source 384W x 224H, landscape):"
echo "  Source aspect 384/224 = 1.714, target 832/480 = 1.733 (+1.1% drift)"
echo "  ImageCropAndResize (cover mode, 保 aspect ratio):"
echo "    scale = max(832/384, 480/224) = 2.167"
echo "    BILINEAR -> 485H x 832W, center_crop -> 480H x 832W"
echo "    上下各裁约 2.7 px，几何几乎无畸变"
echo ""
echo "Size constraint: Wan2.2-TI2V-5B WanVideoVAE38 upsampling_factor=16,"
echo "  combined with DiT patch_size=2 requires H/W divisible by 32."
echo "  480 % 32 == 0, 832 % 32 == 0. ✓"
echo ""
echo "Token budget (patched latent):"
echo "  VAE 16x + DiT patch 2x => factor 32"
echo "  latent_T (26) x H/32 (15) x W/32 (26) = 10,140 patched tokens"
echo ""
echo "Dataset: SF3 clips_10s_10k (39,562 train rows)."
echo "  Training num_frames=101 -> LoadVideo truncates to first 5s."
echo ""
echo "Unchanged vs SF2 baseline: batch=1/GPU, lr 5e-5, save_steps 1000,"
echo "           verbose prompt, prompt_dropout=0.1, action_dropout=0.0,"
echo "           cold start (no --resume_from_ckpt)."
echo ""
echo "Start: $(date)"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

PYTHONUNBUFFERED=1 accelerate launch --num_processes=8 --multi_gpu --main_process_port 29520 \
  examples/ReactiveGWM/model_training/train.py \
  --game sf3 \
  --dataset_base_path /home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k \
  --dataset_metadata_path /home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k/metadata_train.csv \
  --data_file_keys "video,action" \
  --height 480 \
  --width 832 \
  --num_frames 101 \
  --dataset_repeat 1 \
  --model_paths '[["/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],"/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth","/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' \
  --tokenizer_path "/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --learning_rate 5e-5 \
  --num_epochs 1000 \
  --save_steps 1000 \
  --output_path "/nfs/zeqingwang/models/train/sf3/phase1_action/p1_sf3_480x832_5s_baseline" \
  --use_gradient_checkpointing \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --action_hold_window 10 \
  --action_dropout_prob 0.0 \
  --use_csv_prompt \
  --prompt_column prompt \
  --prompt_dropout_prob 0.1 \
  --dataset_num_workers 4

echo ""
echo "Training finished at $(date)"
