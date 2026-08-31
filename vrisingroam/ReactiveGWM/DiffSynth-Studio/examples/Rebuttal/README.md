# ReactiveGWM Rebuttal Experiments

This directory contains three isolated SF2 rebuttal training variants:

| Variant | Prompt | Trainable parameters |
|---|---|---|
| `v1` / `vanilla_strategy` | vanilla narration + one space + `Strategy(...)` | full DiT |
| `v2` / `strategy_only` | `Strategy(...)` only | full DiT |
| `v3` / `hybrid_cross_lora` | original structured prompt | cross-attention LoRA; all other DiT parameters full |

All variants cold-start from Wan2.2-TI2V-5B and keep the per-frame action
input.  No released `Vanilla.safetensors` checkpoint is used.

The authoritative design is [plan.md](plan.md).  Live implementation status is
tracked in [progress.md](progress.md).

## Isolation contract

- Source files outside `examples/Rebuttal/` are read-only dependencies.
- Source metadata files are never rewritten.
- Generated metadata, caches, logs, and outputs are ignored by Git.
- Formal training is not launched until metadata, gradient, and checkpoint
  smoke tests pass.

## Data preparation

```bash
/home/zeqingwang/anaconda3/envs/reactivegwm/bin/python \
  examples/Rebuttal/prepare_metadata.py
```

The default inputs are:

```text
/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF2/metadata_vanilla.csv
/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF2/metadata.csv
```

The generated CSV files are written under `examples/Rebuttal/generated/`.

## Validation

Run the complete lightweight suite:

```bash
/home/zeqingwang/anaconda3/envs/reactivegwm/bin/python -m unittest \
  discover -s examples/Rebuttal/tests -v
```

The suite covers prompt construction, metadata hashes, full/V3 trainable
policies, optimizer membership, exact step stopping, gradient contracts,
checkpoint split/reload/fusion, shared-cache layout, and evaluation prompt
replacement.

Run all three two-step, non-cached GPU smoke tests:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash examples/Rebuttal/launch/smoke_all.sh
```

Each smoke run uses 16 rows, saves at steps 1 and 2, and performs the mandatory
gradient audit. V3 checks LoRA B at step 1 and both LoRA A/B at step 2.

Validate weight-only reload from the step-1 smoke checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 \
  SMOKE_SOURCE_BASE=/path/to/completed/smoke \
  bash examples/Rebuttal/launch/smoke_reload.sh
```

## Formal cache

The cache has one physical VAE store and three independent T5 stores:

```text
${CACHE_BASE}/shared_vae/{video,first_frame}
${CACHE_BASE}/v1/t5
${CACHE_BASE}/v2/t5
${CACHE_BASE}/v3/t5
```

Prepare it with:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  CACHE_VARIANTS=v1 \
  CACHE_BASE=/nfs/zeqingwang/cache/ReactiveGWM/Rebuttal/SF2_480x608x101 \
  bash examples/Rebuttal/launch/prepare_cache.sh
```

Workers shard both T5 prompts and video rows. The finalizer refuses to write a
manifest until every expected video, first-frame, prompt, and empty-prompt
shard exists. Omit `CACHE_VARIANTS` to prepare all three variants.

## Formal training

The three approved 6-GPU launches are:

```bash
bash examples/Rebuttal/launch/train_v1_vanilla_strategy.sh
bash examples/Rebuttal/launch/train_v2_strategy_only.sh
bash examples/Rebuttal/launch/train_v3_hybrid_cross_lora.sh
```

Prepare only V1, validate its cache manifest, and then launch V1 automatically:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  bash examples/Rebuttal/launch/cache_then_train_v1.sh
```

While the canonical V1 workers remain active, optional VAE-only workers can
fill missing rows from the end of the dataset without loading T5 or DiT:

```bash
bash examples/Rebuttal/launch/supplement_v1_cache.sh
```

The default supplemental layout is three low-priority workers on GPU 0 and
one on GPU 1. Atomic claims prevent supplemental workers from duplicating one
another; the canonical workers remain authoritative.

Additional workers can be restricted to a protected tail range. Install
canonical guard claims below the boundary first, then launch uniquely named
workers with the same minimum CSV index:

```bash
python examples/Rebuttal/supplemental_claim_guard.py \
  --metadata examples/Rebuttal/generated/metadata_v1_vanilla_strategy.csv \
  --cache_root "${CACHE_BASE}/v1" \
  --below_csv_index 5000

SUPPLEMENTAL_LAYOUT=gpu2-extra:2,gpu3-extra:3,gpu4-extra:4,gpu5-extra:5,gpu6-extra:6,gpu7-extra:7 \
  SUPPLEMENTAL_MIN_CSV_INDEX=5000 \
  bash examples/Rebuttal/launch/supplement_v1_cache.sh
```

The V1 training launch waits for every supplemental VAE worker to exit before
loading the training model on GPUs 2-7.

Useful environment overrides:

```text
CUDA_VISIBLE_DEVICES  default 2,3,4,5,6,7
CACHE_BASE            shared cache parent
OUTPUT_BASE           parent of all three training outputs
OUT                   output for one selected launch
PORT                  Accelerate port for one selected launch
```

The entry rejects formal configuration drift. Approved values are 30,000
steps, save every 1,000, AdamW `5e-5`, weight decay `0.01`, one optimizer
group, accumulation 1, prompt dropout `0.1`, action dropout `0.0`, repeat 1,
four workers, 480×608×101, and action hold window 10.

V1 and V3 use the approved six-process recipe with effective batch size 6.
V2 uses four processes on GPUs 0,3,4,5 with effective batch size 4.

Every launch writes `run_config.json` before optimization. It contains the
metadata hash, base-model fingerprint, visible GPUs, free disk space, complete
arguments, and parameter audit.

## Checkpoints and resume

V1/V2 write:

```text
step-N.safetensors
```

Resume weights with:

```bash
bash examples/Rebuttal/launch/train_v1_vanilla_strategy.sh \
  --resume_checkpoint /path/to/step-N.safetensors
```

V3 writes:

```text
step-N.full.safetensors
step-N.lora.safetensors
step-N.manifest.json
```

Resume V3 with:

```bash
bash examples/Rebuttal/launch/train_v3_hybrid_cross_lora.sh \
  --resume_manifest /path/to/step-N.manifest.json
```

All resumes are weight-only by default. `--save_full_state` and
`--resume_state /path/to/state-N` explicitly opt into optimizer/RNG state.

V3 can be exported as one standard full DiT:

```bash
/home/zeqingwang/anaconda3/envs/reactivegwm/bin/python \
  examples/Rebuttal/export_merged.py \
  --manifest /path/to/step-N.manifest.json \
  --output /path/to/final_merged.safetensors
```

## Fixed-condition strategy evaluation

The evaluator fixes the source clip, first frame, action parquet, seed, and
sampling parameters, changing only `Strategy(...)`:

```bash
/home/zeqingwang/anaconda3/envs/reactivegwm/bin/python \
  examples/Rebuttal/eval.py \
  --variant v3 \
  --manifest /path/to/step-N.manifest.json \
  --row_index 0 \
  --output_dir /path/to/eval
```

Use `--checkpoint` instead of `--manifest` for V1/V2 or a merged V3
checkpoint.

## Storage warning

Saving all 30 checkpoints for all three variants is expected to consume about
0.75–0.85 TB. No checkpoint is deleted automatically. Formal launches should
only begin after confirming the output filesystem and exclusive GPU
availability.
