#!/bin/bash
# For each new stage1 ckpt (every 500 steps), run tour20s + W5sD5s AR probes on 252 GPU7.
cd ~/vrisingroam && . env.sh
RUN=/data/yuzhewu/vrisingroam/distill/runs/gm_ar_tf_v1
OUT=/data/yuzhewu/vrisingroam/eval_v2ca/stage1_resume_probes; mkdir -p $OUT
for S in 8500 9000 9500 10000 10500 11000 11500 12000; do
  until [ -f $RUN/dits/dit_step$S.safetensors ]; do sleep 600; done
  sleep 60
  for P in tour20s W5s_D5s; do
    CUDA_VISIBLE_DEVICES=7 python scripts/s1_ar_rollout.py \
      --dit $RUN/dits/dit_step$S.safetensors \
      --image /data/yuzhewu/vrisingroam/eval_teacher/bridge_seed.png \
      --actions /data/yuzhewu/vrisingroam/eval_teacher/$P.parquet \
      --table /data/yuzhewu/vrisingroam/processed/action_context_table_v3.pt \
      --out $OUT/s1_${S}_${P}.mp4 --steps 30 --kv_window 19 --rope_cap 20 --sink_size 1 > /dev/null 2>&1
  done
  python - <<PY >> $OUT/summary.txt
from scripts.measure_motion import measure
S=$S
segs=[("W",0,100),("D",100,160),("W2",160,220),("A",220,280),("S",280,340),("rel",340,401)]
p=f"$OUT/s1_{S}_tour20s.mp4"
t=" ".join(f"{n}:{measure(p,start=a,end=b)[1 if n in ('D','A') else 2]:+.0f}" for n,a,b in segs)
p2=f"$OUT/s1_{S}_W5s_D5s.mp4"
w=" ".join(f"{n}:{measure(p2,start=a,end=b)[1 if n.startswith('D') else 2]:+.0f}" for n,a,b in [("W1",0,50),("W2",50,100),("D1",100,150),("D2",150,201)])
print(f"step {S} | tour: {t} | W5sD5s: {w}")
PY
done
