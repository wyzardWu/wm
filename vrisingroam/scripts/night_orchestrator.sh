#!/bin/bash
# Unattended overnight orchestration (runs on node1 via nohup):
#   1. wait for 0731 cache complete on node2
#   2. launch node2-side 0730 cache shards (GPUs 2/3, beside training)
#   3. wait for 0730 fully cut+synced+cached on node2
#   4. merge both sessions into combined_5d (hardlinks, cache reused)
#   5. stop the subset training, relaunch on combined_5d resuming weights
# Every step logs; any hard failure leaves training running untouched.
set -u
N2="ssh -o BatchMode=yes -o ConnectTimeout=20 yuzhewu@h200-node2"
B=/data/yuzhewu/vrisingroam/processed
log() { echo "$(date +%m-%d\ %H:%M) $*"; }

log "orchestrator up"

# ---- 1. 0731 cache complete on node2
while :; do
  c=$($N2 "find $B/20260731_nativecrop/cache/video -name '*.pt' | wc -l" 2>/dev/null || echo 0)
  log "0731 cache: $c/21731"
  [ "${c:-0}" -ge 21731 ] && break
  sleep 300
done
log "0731 cache COMPLETE"

# ---- 2. node2 0730 shards on GPUs 2/3 (skip_existing unions with node1's)
$N2 'cd ~/vrisingroam && . env.sh && OUT=data/processed/20260730_nativecrop && i=0; for g in 2 2 3 3; do CUDA_VISIBLE_DEVICES=$g nohup python -m ReactiveGWM_Code.training.bidirectional.precompute_cache --game vrising --metadata $OUT/metadata.csv --dataset_base $OUT --cache_root $OUT/cache --model_paths "$DIT_JSON" --tokenizer_path "$TOKENIZER" --height 480 --width 832 --num_frames 101 --max_rows 0 --shard_rank $i --shard_world_size 4 --skip_existing --skip_manifest_write > $OUT/precompute_n2_r$i.log 2>&1 & i=$((i+1)); done; echo launched' < /dev/null \
  && log "node2 0730 shards launched" || log "WARN: node2 0730 shard launch failed"

# ---- 3. 0730 fully cut + synced + cached
while :; do
  grep -q 0730_FINAL_SYNC_DONE /data/yuzhewu/vrisingroam/sync_0730.log 2>/dev/null && break
  log "waiting 0730 final sync"; sleep 300
done
ROWS=$($N2 "python3 -c \"import sys; print(sum(1 for _ in open('$B/20260730_nativecrop/metadata.csv'))-1)\"")
log "0730 rows: $ROWS"
while :; do
  c=$($N2 "find $B/20260730_nativecrop/cache/video -name '*.pt' | wc -l" 2>/dev/null || echo 0)
  log "0730 cache: $c/$ROWS"
  [ "${c:-0}" -ge "$ROWS" ] && break
  # keep pushing node1-computed shards over
  rsync -a --exclude '*.tmp' --exclude .lock $B/20260730_nativecrop/cache yuzhewu@h200-node2:$B/20260730_nativecrop/ 2>/dev/null
  sleep 300
done
log "0730 cache COMPLETE"
# rerun one rank-0 pass to write a manifest covering all rows (cheap skim)
$N2 'cd ~/vrisingroam && . env.sh && OUT=data/processed/20260730_nativecrop && CUDA_VISIBLE_DEVICES=2 python -m ReactiveGWM_Code.training.bidirectional.precompute_cache --game vrising --metadata $OUT/metadata.csv --dataset_base $OUT --cache_root $OUT/cache --model_paths "$DIT_JSON" --tokenizer_path "$TOKENIZER" --height 480 --width 832 --num_frames 101 --max_rows 0 --shard_rank 0 --shard_world_size 1 --skip_existing >> $OUT/precompute_manifest.log 2>&1' < /dev/null
log "0730 manifest pass done"

# ---- 4. merge
$N2 'cd ~/vrisingroam && ~/miniconda3/envs/rgwm/bin/python scripts/merge_5d.py' < /dev/null
if [ $? -ne 0 ]; then log "FATAL: merge failed — leaving subset training running"; exit 1; fi
log "merge COMPLETE"

# ---- 5. restart training on combined_5d
LATEST=$($N2 "ls ~/vrisingroam/runs/v1_nativecrop/step-*.safetensors 2>/dev/null | sed 's/.*step-//; s/.safetensors//' | sort -n | tail -1")
if [ -z "$LATEST" ]; then log "FATAL: no checkpoint found; leaving training running"; exit 1; fi
REMAIN=$((30000 - LATEST))
log "stopping subset training at >= step $LATEST; will resume for $REMAIN more steps"
$N2 'pkill -f "bidirectional.trai[n]"' < /dev/null; sleep 30
$N2 'pkill -9 -f "bidirectional.trai[n]"' < /dev/null; sleep 10
$N2 "cd ~/vrisingroam && CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 nohup bash scripts/train_bidirectional.sh data/processed/combined_5d v1_full --resume_from_ckpt runs/v1_nativecrop/step-$LATEST.safetensors --max_train_steps $REMAIN > runs/v1_full_launch.log 2>&1 & echo relaunched" < /dev/null \
  && log "training RELAUNCHED on combined_5d (runs/v1_full, resume step-$LATEST, $REMAIN steps)" \
  || log "FATAL: relaunch command failed"

# ---- state janitor: keep newest 2 full-state dirs per run dir
while :; do
  $N2 'for d in ~/vrisingroam/runs/v1_nativecrop ~/vrisingroam/runs/v1_full; do [ -d "$d" ] || continue; ls -d $d/state-* 2>/dev/null | sed "s/.*state-//" | sort -n | head -n -2 | while read s; do rm -rf "$d/state-$s" && echo "pruned $d/state-$s"; done; done' < /dev/null
  sleep 1800
done
