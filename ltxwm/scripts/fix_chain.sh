#!/bin/bash
# 修复链:坏 checkpoint 替换后重建一切衍生物并重启训练
# 前提:/data/yuzhewu/ltxwm/redl/ltx-2.3-22b-dev.safetensors 已下载完成
set -e
cd ~/vrisingroam && . env.sh
LTX=/data/yuzhewu/ltxwm
NEW=$LTX/redl/ltx-2.3-22b-dev.safetensors
WANT=7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f

echo "[fix] sha256 verify $(date)"
GOT=$(sha256sum "$NEW" | awk '{print $1}')
if [ "$GOT" != "$WANT" ]; then echo "[fix] SHA MISMATCH $GOT"; exit 1; fi
echo "[fix] sha ok"

echo "[fix] stop garbage training"
pkill -f "train_action[.]py" || true
sleep 10

echo "[fix] swap checkpoint"
mv $LTX/ltx-2.3-22b-dev.safetensors $LTX/ltx-2.3-22b-dev.safetensors.corrupt
mv "$NEW" $LTX/ltx-2.3-22b-dev.safetensors

echo "[fix] archive corrupt-era probe outputs"
mkdir -p /data/yuzhewu/vrisingroam/ltx_probes/corrupt_era
mv /data/yuzhewu/vrisingroam/ltx_probes/*.mp4 /data/yuzhewu/vrisingroam/ltx_probes/*.png /data/yuzhewu/vrisingroam/ltx_probes/corrupt_era/ 2>/dev/null || true

echo "[fix] VAE roundtrip gate on GPU1 $(date)"
TERM=dumb CUDA_VISIBLE_DEVICES=1 python $LTX/scripts/vae_gate.py

echo "[fix] rebuild action tables (CPU) $(date)"
python $LTX/scripts/build_ltx_action_tables.py

echo "[fix] wipe corrupt precomp + mark old runs"
rm -rf $LTX/data/train2000_precomp $LTX/data/smoke64_precomp
mv $LTX/runs/abot2000_v1 $LTX/runs/abot2000_v1_corrupt 2>/dev/null || true
mv $LTX/runs/smoke64 $LTX/runs/smoke64_corrupt 2>/dev/null || true

echo "[fix] precompute 2000 start $(date)"
CUDA_VISIBLE_DEVICES=1 python $LTX/LTX-2/packages/ltx-trainer/scripts/process_dataset.py \
  $LTX/data/train2000.jsonl \
  --resolution-buckets 832x480x121 \
  --model-path $LTX/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path $LTX/gemma-3-12b-it \
  --skip-audio \
  --output-dir $LTX/data/train2000_precomp

echo "[fix] inject action ids"
python $LTX/ltxwm/prepare_conditions_actions.py \
  --data_root $LTX/data/train2000_precomp \
  --manifest $LTX/data/train2000.jsonl \
  --actions_root $LTX/data/actions

echo "[fix] launching training $(date)"
TERM=dumb PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
python $LTX/ltxwm/train_action.py --config $LTX/configs/abot2000.yaml
echo "[fix] TRAINING DONE $(date)"
