#!/bin/bash
# usage: s1_probe.sh <dit_path> <tag>  -- runs 4 causal probes on GPU7 headroom
set -e
cd ~/vrisingroam && . env.sh
DIT=$1; TAG=$2
OUT=/data/yuzhewu/vrisingroam/eybx_probes/stage1_$TAG
mkdir -p $OUT
S=/data/yuzhewu/eybxroam/probe_seeds
T=/data/yuzhewu/eybxroam/tables
for job in "ar_const_dungeon:$S/infiniteDungeon.png" "ar_switch_d2h_41:$S/infiniteDungeon.png" "ar_switch_d2h_21:$S/infiniteDungeon.png" "ar_ship_null2d_41:$S/ood_shipPrologueA.png"; do
  name=${job%%:*}; img=${job#*:}
  CUDA_VISIBLE_DEVICES=7 python scripts/s1_ar_rollout.py --dit "$DIT" --image "$img" \
    --actions $S/$name.parquet --table $T/eybx_action_table.pt --scene_table $T/eybx_scene_table.pt \
    --out $OUT/$name.mp4 --steps 30 || echo "PROBE $name FAILED"
done
echo S1PROBES_DONE
