#!/bin/bash
# VAE/T5 cache precompute for the vrising dataset, sharded over 8 GPUs.
# Usage: bash scripts/precompute_cache.sh <out_root> [max_rows]
set -e
cd "$(dirname "$0")/.."
. env.sh
OUT_ROOT=${1:?usage: precompute_cache.sh <processed-out-root> [max_rows]}
MAX_ROWS=${2:-0}
CACHE_ROOT=$OUT_ROOT/cache

# Serialize concurrent invocations (incremental filler + chain watcher).
mkdir -p "$CACHE_ROOT"
exec 9>"$CACHE_ROOT/.lock"
flock 9

# Shard only over GPUs no other job occupies (<10 GiB used), unless the
# caller pins an explicit list (may repeat an index to stack that GPU),
# e.g. GPUS_OVERRIDE="0 5 5" PER_GPU=1.
if [ -n "$GPUS_OVERRIDE" ]; then
  FREE_GPUS="$GPUS_OVERRIDE"
else
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 10000 {print $1}')
fi
NGPU=$(echo $FREE_GPUS | wc -w)
[ "$NGPU" -ge 1 ] || { echo "no free GPUs, aborting"; exit 1; }
# The job is CPU-decode bound (~6% GPU util per process), so stack several
# shard processes per GPU.
PER_GPU=${PER_GPU:-3}
NSHARD=$((NGPU * PER_GPU))
echo "sharding $NSHARD ways over $NGPU free GPUs: $FREE_GPUS (x$PER_GPU each)"
r=0
for g in $(for k in $(seq 1 $PER_GPU); do echo $FREE_GPUS; done | tr ' ' '\n'); do
  EXTRA=""
  [ "$r" != "0" ] && EXTRA="--skip_manifest_write"
  CUDA_VISIBLE_DEVICES=$g python -m ReactiveGWM_Code.training.bidirectional.precompute_cache \
    --game "${GAME:-vrising}" \
    --metadata "$OUT_ROOT/metadata.csv" \
    --dataset_base "$OUT_ROOT" \
    --cache_root "$CACHE_ROOT" \
    --model_paths "$DIT_JSON" \
    --tokenizer_path "$TOKENIZER" \
    --height 480 --width 832 --num_frames 101 \
    --max_rows "$MAX_ROWS" \
    --shard_rank "$r" --shard_world_size "$NSHARD" \
    --skip_existing $EXTRA &> "$OUT_ROOT/precompute_rank$r.log" &
  r=$((r + 1))
done
wait
# rank 0 writes the manifest last so it covers all rows; rerun rank 0 alone if
# other shards finished after it.
echo "precompute done; check $OUT_ROOT/precompute_rank*.log"
