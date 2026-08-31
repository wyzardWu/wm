#!/bin/bash
# At true-48.5k (step-91000): bridge+village 20s GT-action chains (cfg1, v3 table) on 252
# GPU7, then splice GEN(hud) | GT side-by-sides. Never touches node2 training.
set -eo pipefail
cd "$(dirname "$0")/.."
. env.sh
set -u
CKPT=/data/yuzhewu/vrisingroam/eval_ckpts/step-91000.safetensors
OUT=/data/yuzhewu/vrisingroam/eval_v2ca/gt20s_50k
TBL=/data/yuzhewu/vrisingroam/processed/action_context_table_v3.pt

rsync -a yuzhewu@h200-node2:~/vrisingroam/runs/v2_ca_bidir_p3/step-91000.safetensors \
      /data/yuzhewu/vrisingroam/eval_ckpts/

for S in bridge village; do
  CUDA_VISIBLE_DEVICES=7 python scripts/eval_chain.py \
    --ckpt "$CKPT" \
    --image /data/yuzhewu/vrisingroam/eval_teacher/${S}_seed.png \
    --name ${S}_gt20s_48k --out_dir "$OUT" --windows 4 \
    --full_actions /data/yuzhewu/vrisingroam/eval_teacher/${S}_gt401.parquet \
    --action_cfg 1.0 --action_context_table "$TBL"
done

python - <<'PYEOF' > "$OUT/motion_summary_48k.txt"
from scripts.measure_motion import measure
for s in ['bridge','village']:
    v=f'/data/yuzhewu/vrisingroam/eval_v2ca/gt20s_50k/{s}_gt20s_48k_chain4.mp4'
    g=f'/data/yuzhewu/vrisingroam/eval_v2ca/gt20s_50k/{s}_GT20s.mp4'
    for tag,path in [('GEN',v),('GT ',g)]:
        out=[]
        for w in range(4):
            m,dx,dy,n = measure(path, start=101*w, end=101*w+101)
            out.append(f"w{w+1} dx={dx:+.1f} dy={dy:+.1f}")
        print(f"{s:8s} {tag} 48k:", " | ".join(out), flush=True)
PYEOF

for S in bridge village; do
  ffmpeg -y -v error \
    -i "$OUT/${S}_gt20s_48k_chain4_hud.mp4" -i "$OUT/${S}_GT20s.mp4" \
    -filter_complex "[0:v]drawtext=text='GEN v3@48k cfg1':x=10:y=10:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5[l];[1:v]drawtext=text='GT':x=10:y=10:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5[r];[l][r]hstack" \
    -c:v libx264 -crf 20 "$OUT/${S}_48k_vsGT.mp4"
done
echo ALL_DONE
