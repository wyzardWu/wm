# Rebuttal Implementation Progress

This file is the compact-safe progress anchor for
[plan.md](plan.md).  When context is compacted, resume from the first
incomplete item below and verify the recorded commands before continuing.

Last updated: 2026-07-24

## Locked decisions

- All three variants cold-start from raw Wan2.2.
- V1 prompt is `vanilla.rstrip() + " " + Strategy(...)`.
- V2 prompt is exactly `Strategy(...)`.
- All variants keep the action parquet input.
- V1 and V2 full-fine-tune the complete DiT.
- V3 freezes the original cross-attention branch, including `norm3`, and adds
  rank-32 LoRA to cross-attention q/k/v/o.
- V3 full-fine-tunes every other DiT parameter.
- Every trainable parameter uses one AdamW optimizer at learning rate `5e-5`.
- V1/V3 formal launches use six processes with effective batch size 6; V2
  uses four processes on GPUs 0,3,4,5 with effective batch size 4.
- All new source changes stay under `examples/Rebuttal/`.

## Status

| Stage | State | Evidence |
|---|---|---|
| Directory skeleton and source isolation | complete | README, package markers, local ignore rules, and this compact-safe tracker created |
| Derived metadata implementation | complete | 10,000-row V1/V2 outputs generated twice with stable hashes; 7 unit tests pass; UMT5 maxima are 122/27/305 |
| V1/V2 full-DiT training | implementation complete | Unified entry applies and audits full-DiT training; three smoke preflights pass |
| V3 hybrid LoRA training | implementation complete | Real 30-layer meta audit proves 120 LoRA modules / 240 tensors / 23,592,960 scalars |
| Runner and checkpoint contract | implementation complete | Exact-step runner, one AdamW group, V1/V2 full saves, V3 split saves, resume, hashes, and merge exporter tested |
| Shared VAE / separate T5 cache | all three formal caches complete | V2/V3 reuse all 10,000 shared VAE rows and have separate, finalized T5 caches |
| Smoke and inference validation | complete | Three real 5B two-step runs, weight-only reloads, V3 merge, and fixed-condition inference passed |
| Formal cache and 30k training | V1 and V2 launched; V3 cache ready | V2 formal four-GPU optimizer is active on GPUs 0,3,4,5; V3 formal preflight passes against its finalized cache |

## Completed evidence

- V1 SHA256:
  `7a5b7d2bbd8baad3249074024dd133b05ebac6559a98045758ce7e920db6de16`
- V2 SHA256:
  `6fc14c581610c61eb9e50c3aa6a5067e6962f6ac09ede33854993f8b7215f5fa`
- Metadata tests:
  `/home/zeqingwang/anaconda3/envs/reactivegwm/bin/python -m unittest examples.Rebuttal.tests.test_metadata -v`
  passed 7/7.
- Syntax checks for the generator and tests passed.
- Re-running `prepare_metadata.py` produced the same output hashes.
- The `reactivegwm` environment does not contain `black`; the host
  installation is `/home/zeqingwang/anaconda3/bin/black`.
- Full lightweight test command:
  `/home/zeqingwang/anaconda3/envs/reactivegwm/bin/python -m unittest discover -s examples/Rebuttal/tests -v`
  passed 41/41 after adding the T5-only worker CLI test.
- Host `black --check`, Black-compatible
  `flake8 --max-line-length=88 --extend-ignore=E203`, Python compilation,
  shell `bash -n`, and `git diff --check` all pass.
- V1/V2/V3 smoke-mode preflight passed with the correct 16-row metadata,
  prompt mode, `video,action` fields, and exact two-step settings.
- Actual ReactiveGWM meta-device V3 audit:
  - total DiT + LoRA: 5,024,302,272 scalars;
  - frozen original cross branch + norm3: 1,133,199,360 scalars;
  - full-finetuned non-cross parameters: 3,867,509,952 scalars;
  - cross LoRA: 23,592,960 scalars;
  - total trainable: 3,891,102,912 scalars;
  - action embedders: 921,600 trainable scalars.
- Real smoke artifact roots:
  - initial runs:
    `/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/smoke_validation_20260724`;
  - reload runs:
    `/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/smoke_validation_20260724_reload`.
- Two-step smoke losses:
  - V1: `0.091223`, `0.215707`;
  - V2: `0.092588`, `0.214392`;
  - V3: `0.174923`, `0.439282`.
- V1/V2 checkpoints each contain all 855 DiT keys. V3 full delta contains
  exactly 495 keys / 3,867,509,952 scalars; V3 LoRA contains exactly 240 keys
  / 23,592,960 scalars.
- Gradient audits:
  - V1/V2 action, self-attn, cross-attn, and FFN are finite/nonzero;
  - V3 action, self-attn, FFN, and LoRA B are finite/nonzero at step 1;
  - V3 LoRA A and B are both finite/nonzero at step 2;
  - frozen V3 original cross-attn and `norm3` received no gradients.
- Weight-only reload:
  - V1/V2 loaded 855 keys with `missing=0, unexpected=0`;
  - V3 loaded 495 full keys with the expected 360 frozen keys missing and
    `unexpected=0`, then loaded all 240 LoRA keys;
  - all three reloads continued from step 1 to step 2.
- Exact V3 merged export:
  - 855 keys / 5,000,709,312 BF16 scalars;
  - SHA256
    `1de2e5f055c2b062f8d0b8a9b80fa0e2d4b680977c37441d64a8dc0542767bc8`.
- Split-versus-merged fixed inference:
  - both MP4 files have SHA256
    `2c1f0438eafe56aecce5a082b526963a446782bfa0c679a5654d827f8bcf403f`;
  - all 88,427,520 decoded pixel-channel values match; maximum error is 0.
- Real cache-worker smoke:
  - artifact root:
    `/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/cache_smoke_validation_20260724`;
  - encoded 9 unique T5 prompts and 16 video/first-frame rows with no failures;
  - the exhaustive finalizer wrote a valid 16-row manifest bound to the V2
    smoke metadata SHA256;
  - cached tensors are BF16 with video latent shape `[1, 48, 25, 30, 38]`
    and first-frame latent shape `[1, 48, 1, 30, 38]`;
  - `video` and `first_frame` resolve to the shared VAE store while `t5`
    remains variant-specific;
  - the smoke cache occupies approximately 80 MiB and loaded T5/VAE only,
    without loading the DiT.
- V1, V2, and V3 each generated two fixed-condition 101-frame videos where
  only `Strategy(...)` changed. V1 kept vanilla narration fixed, V2 contained
  only Strategy, and V3 kept Active/Passive fixed.
- Source metadata hashes remain unchanged after all smoke and inference runs.
- The V1 formal 30,000-step optimizer is active on GPUs 2-7. V2 and V3 formal
  optimizer processes have not started.
- V1 formal cache-to-training orchestration PID: `560176`.
- Orchestration log:
  `/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/v1_cache_then_train_20260724T081807Z.log`.
- V1 canonical cache workers `561732` through `561737` completed with world
  size 6 on GPUs 2-7.
- Supplemental manager PID `852355` launched workers `852363` through
  `852366` at 2026-07-24 08:38 UTC: three on GPU 0 and one on GPU 1.
- Supplemental workers are VAE-only, reverse-scanning, low-priority, limited
  to six CPU threads each, and coordinate through atomic claims.
- Five consecutive 50-second windows sustained 3.28-3.60 videos/second versus
  the 1.14 videos/second baseline, with zero supplemental failures.
- Supplemental orchestration log:
  `/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/v1_supplemental_20260724T083816Z.log`.
- A canonical boundary guard reserves incomplete CSV indices below 5,000 for
  the six original forward workers. At installation it created 2,889 guard
  claims and found 2,111 lower-half rows already complete.
- Supplemental manager PID `1503205` launched one additional VAE-only worker
  on each of GPUs 2-7 (`1503213` through `1503218`). These workers are
  explicitly bounded to CSV indices 5,000-9,999.
- All ten supplemental workers shared atomic per-video claims. A live audit
  during the cache run found exactly 2,889 boundary guards plus ten active
  worker claims.
- Post-expansion windows measured 7.32 and 6.17 videos/second with zero
  failures. The training entry now waits for all supplemental VAE workers to
  exit before loading the six-GPU training model.
- GPU2-7 supplemental orchestration log:
  `/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/v1_supplemental_gpu2_7_20260724T085402Z.log`.
- V2 formal T5 cache completed at 2026-07-24 10:01 UTC:
  - eight T5-only workers ran one per physical GPU 0-7;
  - 10 unique embeddings were written: nine Strategy prompts plus empty;
  - all eight reports recorded zero failures;
  - the V2 manifest covers 10,000 rows and reuses the V1 shared video and
    first-frame VAE directories;
  - metadata SHA256 is
    `6fc14c581610c61eb9e50c3aa6a5067e6962f6ac09ede33854993f8b7215f5fa`;
  - formal V2 cached-training preflight passed for six GPUs.
- V2 T5 orchestration log:
  `/nfs/zeqingwang/cache/ReactiveGWM/Rebuttal/SF2_480x608x101/v2/t5-orchestrator-20260724T100005Z.log`.
- V3 formal T5 cache completed at 2026-07-24 10:11 UTC:
  - eight T5-only workers ran one per physical GPU 0-7;
  - 4,846 unique embeddings were written: 4,845 structured prompts plus empty;
  - all eight reports recorded zero failures;
  - the V3 manifest covers 10,000 rows and reuses the shared video and
    first-frame VAE directories;
  - metadata SHA256 is
    `4e9c3fcfa934cc2308a696d7391ec2f29e50b2e759f5888398a727f98585362b`;
  - formal V3 cached-training preflight passed for six GPUs and reports the
    approved rank-32 cross-attention LoRA configuration.
- V3 T5 orchestration log:
  `/nfs/zeqingwang/cache/ReactiveGWM/Rebuttal/SF2_480x608x101/v3/t5-orchestrator-20260724T100910Z.log`.
- V2 formal training launched on GPUs 0,3,4,5 with four processes and
  effective batch size 4:
  - environment: `/home/zeqingwang/anaconda3/envs/diffsynth`;
  - persistent tmux session: `rebuttal_v2_20260724`;
  - log:
    `/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/v2_strategy_only_20260724T115200Z.log`;
  - checkpoints and run artifacts:
    `/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/v2_strategy_only`;
  - the step-1 gradient audit passed for action, self-attention,
    cross-attention, and FFN parameters.

## Next action

Monitor the launched V1 and V2 30,000-step jobs. V3 is cache-complete and
ready for its formal six-GPU launch when requested.
