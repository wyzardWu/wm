#!/bin/bash
# Probe each new eybx CD EMA ckpt: 2-step AR rollout, const + mid-switch.
cd ~/vrisingroam && . env.sh
EMAS=/data/yuzhewu/vrisingroam/distill/runs/eybx_cd_v1/ema
S=/data/yuzhewu/eybxroam/probe_seeds; T=/data/yuzhewu/eybxroam/tables
done_list=/data/yuzhewu/eybxroam/cd_probed.txt
touch $done_list
while true; do
  for e in $(ls $EMAS/ema_step*.safetensors 2>/dev/null); do
    s=$(basename $e | sed 's/ema_step\([0-9]*\).safetensors/\1/')
    grep -q "^$s$" $done_list && continue
    O=/data/yuzhewu/vrisingroam/eybx_probes/cd_ema$s
    mkdir -p $O
    for job in "ar_const_dungeon:$S/infiniteDungeon.png" "ar_switch_d2h_41:$S/infiniteDungeon.png"; do
      name=${job%%:*}; img=${job#*:}
      CUDA_VISIBLE_DEVICES=2 python scripts/s1_ar_rollout.py --dit "$e" --image "$img" \
        --actions $S/$name.parquet --table $T/eybx_action_table.pt --scene_table $T/eybx_scene_table.pt \
        --out $O/$name.mp4 --steps 2 >> /data/yuzhewu/eybxroam/cd_probe_loop.log 2>&1 || echo "FAIL $s $name" >> /data/yuzhewu/eybxroam/cd_probe_loop.log
    done
    echo "$s" >> $done_list
  done
  sleep 300
done
