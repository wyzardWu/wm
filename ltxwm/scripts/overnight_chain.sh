#!/bin/bash
# 过夜链:等 表编码完成 + precompute 全量完成 → sidecar 注入 → 报告
set -u
LTX=/data/yuzhewu/ltxwm
until grep -aq 'WW_TABLES_DONE' $LTX/ww_tables.log 2>/dev/null; do sleep 300; done
echo "[chain] tables done $(date)"
until grep -aqE 'preprocessing complete' $LTX/precomp_ww.log 2>/dev/null && ! pgrep -f 'process_dataset.py data/ww_pilot' >/dev/null; do sleep 600; done
echo "[chain] precompute done $(date)"
/home/yuzhewu/miniconda3/envs/rgwm/bin/python $LTX/ltxwm/ww_prepare_sidecar.py \
  --precomp $LTX/data/ww_pilot_precomp --workers 16 >> $LTX/ww_sidecar.log 2>&1
echo "[chain] sidecar done $(date): $(grep -a SIDECAR_DONE $LTX/ww_sidecar.log | tail -1)"
echo "OVERNIGHT_CHAIN_DONE $(date)"
