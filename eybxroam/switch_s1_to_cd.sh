#!/bin/bash
# Auto-switch: stage1 -> stage2 CD at step 4000 (user 8/26: "4000就stage2吧")
set -e
until [ -f /data/yuzhewu/vrisingroam/distill/runs/eybx_ar_tf_v1/dits/dit_step4000.safetensors ]; do sleep 120; done
echo "[switch] step-4000 saved; stopping stage1"
for p in $(pgrep -f "ar_diffusion_g[m]"); do kill "$p" 2>/dev/null; done
cd ~/vrisingroam && . env.sh
for g in 2 3 5 7; do HOLD_HEADROOM_MIB=15000 nohup python scripts/hold_eybx.py $g > /data/yuzhewu/eybxroam/hold$g.log 2>&1 & done
sleep 20
for p in $(pgrep -f "ar_diffusion_g[m]"); do kill -9 "$p" 2>/dev/null; done
echo "[switch] launching stage2 CD"
STEP=4000 nohup bash scripts/launch_stage2_eybx.sh > /data/yuzhewu/eybxroam/stage2.log 2>&1 &
echo "[switch] done"
