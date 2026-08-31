#!/bin/bash
# Probe every new stage1 ckpt with the 4 causal scene-switch probes.
cd ~/vrisingroam && . env.sh
DITS=/data/yuzhewu/vrisingroam/distill/runs/eybx_ar_tf_v1/dits
done_list=/data/yuzhewu/eybxroam/s1_probed.txt
touch $done_list
while true; do
  for d in $(ls $DITS/dit_step*.safetensors 2>/dev/null); do
    s=$(basename $d | sed 's/dit_step\([0-9]*\).safetensors/\1/')
    grep -q "^$s$" $done_list && continue
    echo "$(date '+%T') probing step $s" >> /data/yuzhewu/eybxroam/s1_probe_loop.log
    bash /data/yuzhewu/eybxroam/s1_probe.sh "$d" "step$s" >> /data/yuzhewu/eybxroam/s1_probe_loop.log 2>&1
    echo "$s" >> $done_list
  done
  sleep 300
done
