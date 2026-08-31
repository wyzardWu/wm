#!/bin/bash
# 等 5996 条 precompute 完成 → 注入 action_ids → 校验 → 停 2000-clip run → 无缝起 8k run
set -u
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python

until grep -aq 'preprocessing complete' $LTX/precomp8k.log 2>/dev/null; do sleep 120; done
echo "[switch] precompute done $(date)"

$PY $LTX/ltxwm/prepare_conditions_actions.py \
  --data_root $LTX/data/train8k_precomp \
  --manifest $LTX/data/train8k_minus4.jsonl \
  --actions_root $LTX/data/actions
echo "[switch] action ids injected $(date)"

NCOND=$(find $LTX/data/train8k_precomp/conditions -name '*.pt' | wc -l)
NLAT=$(find $LTX/data/train8k_precomp/latents -name '*.pt' | wc -l)
echo "[switch] conditions=$NCOND latents=$NLAT (expect 7996)"
if [ "$NCOND" -lt 7996 ] || [ "$NLAT" -lt 7996 ]; then
  echo "[switch] COUNT MISMATCH, abort"; exit 1
fi

echo "[switch] stopping abot20k run $(date)"
pkill -f 'abot20k.yaml' || true
sleep 45

echo "[switch] launching abot8k $(date)"
cd $LTX
TERM=dumb TMPDIR=/data/yuzhewu/tmp TRITON_CACHE_DIR=$LTX/triton_cache \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3,5,7 \
nohup /home/yuzhewu/miniconda3/envs/rgwm/bin/accelerate launch --num_processes 4 --multi_gpu --main_process_port 29618 \
  $LTX/ltxwm/train_action.py --config $LTX/configs/abot8k.yaml \
  > $LTX/ddp4_8k.log 2>&1 &
echo "[switch] abot8k pid $! $(date)"

# 确认起步
for i in $(seq 1 60); do
  if grep -aq '\[step 1\]' $LTX/ddp4_8k.log 2>/dev/null; then echo "[switch] ABOT8K_STEPPING $(date)"; exit 0; fi
  if grep -aqE 'Error|Traceback|out of memory' $LTX/ddp4_8k.log 2>/dev/null; then echo "[switch] LAUNCH_ERROR $(date)"; exit 1; fi
  sleep 30
done
echo "[switch] TIMEOUT waiting for step 1"; exit 1
