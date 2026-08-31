#!/bin/bash
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "=== SF3 Phase 1 at 480x832, 5s, CACHED (T5+VAE offline) ==="
echo ""
echo "Motivation: skip T5 encode + VAE encode per step (mirror of SF2 cached path)."
echo "  Expected savings: ~15% wall-time, ~22GB/GPU memory (VAE + T5 unloaded)."
echo ""
echo "Cache contract:"
echo "  * Precompute must use UnifiedDataset.default_video_operator(H=480, W=832,"
echo "    num_frames=101, hdf=16, wdf=16) — BILINEAR scale-then-center-crop via"
echo "    ImageCropAndResize (cover mode, 保 aspect ratio，不拉伸)."
echo "  * Bit-exact loss parity gate (|Δloss| < 1e-2 vs non-cached) must pass"
echo "    before switching this run into the main SF3 training line."
echo ""
echo "Resolution design (SF3 source 384W x 224H, landscape):"
echo "  Source aspect: 384/224 = 1.714"
echo "  Target 480x832 (HxW), aspect 832/480 = 1.733, +1.1% drift"
echo "  scale = max(832/384, 480/224) = max(2.167, 2.143) = 2.167"
echo "  BILINEAR upsample -> (485.5H, 832W), center_crop -> (480H, 832W)"
echo "  => 上下各裁约 2.7 px，几何几乎无畸变"
echo ""
echo "Dataset: SF3 clips_10s_10k (39,562 train rows, 120 test rows pre-split)."
echo "  10s @ 20fps source, num_frames=101 -> LoadVideo truncates to first 5s."
echo ""
echo "Precompute prerequisite (run FIRST on spare GPU(s)):"
echo "  CUDA_VISIBLE_DEVICES=0 python examples/ReactiveGWM/scripts/precompute_cache.py \\"
echo "    --csv_path      /home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k/metadata_train.csv \\"
echo "    --dataset_base  /home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k \\"
echo "    --cache_dir     /home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k/cache_480x832_5s \\"
echo "    --height 480 --width 832 --num_frames 101 \\"
echo "    --model_paths <same JSON as --model_paths below> \\"
echo "    --tokenizer_path <same as below> \\"
echo "    --use_csv_prompt --prompt_column prompt \\"
echo "    --skip_existing --verify_first 20"
echo "  (Use --shard_rank / --shard_world_size + multiple tmux sessions to parallelize.)"
echo ""
echo "Start: $(date)"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

PYTHONUNBUFFERED=1 accelerate launch --num_processes=8 --multi_gpu --main_process_port 29521 \
  examples/ReactiveGWM/model_training/train.py \
  --game sf3 \
  --dataset_base_path /home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k \
  --dataset_metadata_path /home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k/metadata_train.csv \
  --use_cached_dataset \
  --cache_root /home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k/cache_480x832_5s \
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
  --output_path "/nfs/zeqingwang/models/train/sf3/phase1_action/p1_sf3_480x832_5s_cached" \
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
