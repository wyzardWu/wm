#!/bin/bash
# Every 1000 CD-v2 steps: probe EMA weights on 252 GPU7 (2-step tour, sliding 19/20/1), log brightness+motion.
cd ~/vrisingroam && . env.sh
RUN=/data/yuzhewu/vrisingroam/distill/runs/gm_cd_v2
OUT=/data/yuzhewu/vrisingroam/eval_v2ca/stage2v2_probes; mkdir -p $OUT
for S in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000; do
  until [ -f $RUN/ema/ema_step$S.safetensors ]; do sleep 900; done
  sleep 90
  CUDA_VISIBLE_DEVICES=7 python scripts/s1_ar_rollout.py \
    --dit $RUN/ema/ema_step$S.safetensors \
    --image /data/yuzhewu/vrisingroam/eval_teacher/bridge_seed.png \
    --actions /data/yuzhewu/vrisingroam/eval_teacher/tour20s.parquet \
    --table /data/yuzhewu/vrisingroam/processed/action_context_table_v3.pt \
    --out $OUT/cdv2_ema${S}_tour20s_steps2.mp4 --steps 2 --kv_window 19 --rope_cap 20 --sink_size 1 > /dev/null 2>&1
  python - <<PY >> $OUT/summary.txt
import imageio.v3 as iio, numpy as np
from scripts.measure_motion import measure
S=$S; p=f"$OUT/cdv2_ema{S}_tour20s_steps2.mp4"
fr=iio.imread(p); lum=fr.astype(np.float32).mean(axis=(1,2,3))
segs=[("W",0,100),("D",100,160),("W2",160,220),("A",220,280),("S",280,340),("rel",340,401)]
mo=" ".join(f"{t}:{measure(p,start=a,end=b)[1 if t in ('D','A') else 2]:+.0f}" for t,a,b in segs)
print(f"ema{S} | 亮 {lum.mean():.1f} ({lum[:20].mean():.0f}->{lum[-20:].mean():.0f}) | {mo}")
PY
done
