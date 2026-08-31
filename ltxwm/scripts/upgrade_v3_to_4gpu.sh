#!/bin/bash
# 监视 GPU2/3:他人占用都降到 <25G 时,把 v3 从 2 卡(5,7)无缝升到 4 卡(2,3,5,7)。
# 升级 = kill 当前 → accum 改 1 → 重启(trainer 从 output_dir 的完整 training_state 自动续)。
set -u
LTX=/data/yuzhewu/ltxwm
LOG=$LTX/upgrade_v3.log
free_check() {
  for g in 2 3; do
    used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i $g 2>/dev/null | awk -F', ' '{s+=$2} END {print s+0}')
    [ "$used" -ge 25000 ] && return 1
  done
  return 0
}
until free_check; do sleep 300; done
echo "[upgrade] GPU2/3 free $(date)" >> $LOG

# 至少已有一个完整 state 才升级(否则续训丢进度)
until ls $LTX/runs/abot8k_v3/checkpoints/training_state_step_*.pt >/dev/null 2>&1; do sleep 300; done

for p in $(pgrep -f 'train_action.py'); do kill $p 2>/dev/null; done
sleep 45
sed -i 's/gradient_accumulation_steps: 2/gradient_accumulation_steps: 1/' $LTX/configs/abot8k_v3.yaml
echo "[upgrade] relaunching 4-gpu $(date)" >> $LOG
cd $LTX
TERM=dumb TMPDIR=/data/yuzhewu/tmp TRITON_CACHE_DIR=$LTX/triton_cache \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3,5,7 \
nohup /home/yuzhewu/miniconda3/envs/rgwm/bin/accelerate launch --num_processes 4 --multi_gpu --main_process_port 29619 \
  $LTX/ltxwm/train_action.py --config $LTX/configs/abot8k_v3.yaml \
  --tables $LTX/tables/ltx_action_tables_v3.pt \
  >> $LTX/ddp_v3.log 2>&1 &
NEWPID=$!
for i in $(seq 1 60); do
  if tail -c 200000 $LTX/ddp_v3.log | grep -aq 'Resumed\|resume\|\[step'; then
    echo "[upgrade] 4-gpu stepping $(date)" >> $LOG; exit 0
  fi
  sleep 30
done
echo "[upgrade] 4-gpu FAILED to step, reverting to 2-gpu $(date)" >> $LOG
kill $NEWPID 2>/dev/null; sleep 30
sed -i 's/gradient_accumulation_steps: 1/gradient_accumulation_steps: 2/' $LTX/configs/abot8k_v3.yaml
CUDA_VISIBLE_DEVICES=5,7 nohup /home/yuzhewu/miniconda3/envs/rgwm/bin/accelerate launch --num_processes 2 --multi_gpu --main_process_port 29619 \
  $LTX/ltxwm/train_action.py --config $LTX/configs/abot8k_v3.yaml \
  --tables $LTX/tables/ltx_action_tables_v3.pt \
  >> $LTX/ddp_v3.log 2>&1 &
echo "[upgrade] reverted, pid $! $(date)" >> $LOG
