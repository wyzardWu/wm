#!/bin/bash
# Chain-robustness watcher: every new +2k true steps, pull latest p3 ckpt from node2,
# run bridge+manor gt15s 3-window chains (cfg1, v3 table) on GPU7, log per-window motion.
set -u
cd "$(dirname "$0")/.."
. env.sh
OUT=/data/yuzhewu/vrisingroam/eval_v2ca/chain_watch
LOG=$OUT/chain_watch.log
mkdir -p $OUT
LAST=65000   # micro of last evaluated ckpt (35.5k true)
while true; do
  L=$(ssh -o BatchMode=yes -o ConnectTimeout=20 yuzhewu@h200-node2 \
      'ls ~/vrisingroam/runs/v2_ca_bidir_p3/step-*.safetensors 2>/dev/null | sed "s/.*step-//;s/\.safetensors//" | sort -n | tail -1' </dev/null 2>/dev/null)
  if [ -n "${L:-}" ] && [ "$L" -ge $((LAST + 4000)) ]; then   # +4000 micro = +2k true
    TRUE=$((L/2+3000))
    rsync -a yuzhewu@h200-node2:~/vrisingroam/runs/v2_ca_bidir_p3/step-$L.safetensors /data/yuzhewu/vrisingroam/eval_ckpts/ || { sleep 300; continue; }
    for SCENE in bridge manor; do
      CUDA_VISIBLE_DEVICES=7 python scripts/eval_chain.py \
        --ckpt /data/yuzhewu/vrisingroam/eval_ckpts/step-$L.safetensors \
        --image /data/yuzhewu/vrisingroam/eval_teacher/${SCENE}_seed.png \
        --name ${SCENE}_${TRUE} --out_dir $OUT --windows 3 \
        --full_actions /data/yuzhewu/vrisingroam/eval_v2ca/battery32k/${SCENE}_gt15s_actions_cat.parquet \
        --action_cfg 1.0 \
        --action_context_table /data/yuzhewu/vrisingroam/processed/action_context_table_v3.pt \
        >> $OUT/runs.log 2>&1
      python - "$OUT/${SCENE}_${TRUE}_chain3.mp4" "$SCENE true=$TRUE" >> $LOG <<'PYEOF'
import sys
from scripts.measure_motion import measure
v, tag = sys.argv[1], sys.argv[2]
out=[]
for w in range(3):
    m,dx,dy,n = measure(v, start=101*w, end=101*w+101)
    out.append(f"w{w+1} dx={dx:+.1f} dy={dy:+.1f}")
print(tag, "|", " | ".join(out), flush=True)
PYEOF
    done
    LAST=$L
  fi
  sleep 900
done
