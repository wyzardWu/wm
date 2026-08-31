# ReactiveGWM Causal Forcing (`examples/ReactiveGWM_casual_forcing/`)

把双向 SF3 游戏世界模型 (`ReactiveGWM_base.safetensors`) 通过 **Causal Forcing++ 三阶段蒸馏** 改造为 **4 步 / 逐帧自回归、KV-cache 实时** 因果生成器。

- **状态 (2026-06-06):** Stage 1/2/3 均已训练并验证通过（`sf3_casual_forcing_2` lineage）。Stage 3 长程 rollout 延伸 run (`stage3_dmd_long_from11200`) 仍在进行中。
- 设计文档：[PLAN.md](./PLAN.md) ｜ 实现进度 / run-by-run：[implement.md](./implement.md) ｜ Stage 3 CF 对齐审计：[stage3_cf_alignment_review.md](./stage3_cf_alignment_review.md)
- 唯一算法源：[thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)
- 与 `examples/ReactiveGWM_self_forcing/` **完全独立**（不读、不引用、不复用）

> **谱系说明：** 当前 canonical run 全部在 `sf3_casual_forcing_2/`（Stage 1 用官方 CF++ 对齐修正版从 base 重训得到，见 `configs/stage1_ar.yaml` 头注）。旧的 `sf3_casual_forcing/` 谱系（pre-alignment）已被取代，对应配置在 `configs/archive/*_8card.yaml` / `*_4card.yaml`。

## 1. 最终产物 & 路径

所有产物 step-N 即 EMA 权重（Stage 2/3 save 时导出 EMA）。canonical 输出目录见各 stage yaml 的 `output_path`。

| Stage | 配置 | 最终 ckpt（截至 2026-06-06） |
|---|---|---|
| 1 AR-TF | `configs/stage1_ar.yaml` | `…/sf3_casual_forcing_2/stage1_ar/step-24000.safetensors` |
| 2 CD | `configs/stage2_cd.yaml` | `…/sf3_casual_forcing_2/stage2_cd/step-20000.safetensors` |
| 3 DMD (26-window) | `configs/stage3_dmd.yaml` | `…/sf3_casual_forcing_2/stage3_dmd/step-11400.safetensors` |
| 3 DMD (long rollout, **进行中**) | `configs/stage3_dmd_long_from11200.yaml` | `…/sf3_casual_forcing_2/stage3_dmd_long_from11200/step-4000.safetensors` |

绝对前缀：`/home/zeqingwang/zeqingwang/models/train/`（= `/nfs/zeqingwang/models/train/` 的符号链接，两者等价）。

Stage 3 若跑到 `max_steps` 收尾，runner 还会额外写三件套（`runner_dmd.py:438/446/454`）：`stage3_dmd_generator.safetensors`、`stage3_dmd_ema.safetensors`、`stage3_dmd_critic.safetensors`。目前两个 Stage 3 run 都靠早停 / 仍在跑，未触发 final-products 导出，推理直接用最新 `step-N.safetensors`（已是 EMA 权重）。

## 2. 复现 Recipe

每个 stage 的下一阶段 `student_init` 默认指向上一阶段产物（已写进 canonical yaml；也可用 `STUDENT_INIT=` 覆盖）。launcher 默认 `NUM_PROCESSES=8`，4 卡用 `NUM_PROCESSES=4`。

```bash
# ---- Stage 1 AR-TF (4 卡 DDP, grad_accum=2 → 全局 batch 8) ----
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash launch/stage1_ar.sh
# resume:
RESUME_STATE=…/sf3_casual_forcing_2/stage1_ar/state-24000 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash launch/stage1_ar.sh

# ---- Stage 2 CD (4 卡 DDP, grad_accum=1 → 全局 batch 4) ----
# 默认 student_init/teacher = Stage 1 step-24000
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash launch/stage2_cd.sh

# ---- Stage 3 DMD (FSDP; 全局 batch = 卡数) ----
# 默认 student_init = Stage 2 step-20000；teacher/critic = base_ckpt
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash launch/stage3_dmd.sh
# 长程 rollout 延伸 run（从某个 generator step 起新开 run）:
CFG=$(pwd)/configs/stage3_dmd_long_from11200.yaml \
    CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash launch/stage3_dmd.sh
# resume Stage 3（注意 cf_resume_meta.json 拓扑要匹配 NUM_PROCESSES）:
RESUME_STATE=…/stage3_dmd_long_from11200/state-4000 \
    CFG=$(pwd)/configs/stage3_dmd_long_from11200.yaml \
    CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash launch/stage3_dmd.sh

# ---- 离线 sanity（训练机 OOM 时兜底, 在共享卡上批量出片）----
bash launch/test_stage1.sh --steps latest        # Stage 1 ckpt 批量
bash launch/test_stage3.sh --steps latest        # Stage 3 ckpt 批量

# ---- CPU verify（无需 GPU, 任何改动后必跑）----
python scripts/verify.py state_dict
python scripts/verify.py tf_mask
python scripts/verify.py kv_cache_rollout
```

## 3. 已知陷阱

- **训练中 sanity OOM**：训练把卡占满后，spawn 的 sanity subprocess（受 `sanity_sample_memory_fraction` 限额）可能 OOM，`samples/step_N.log` 只剩 traceback、没有 mp4。用 `launch/test_stage{1,3}.sh` 在共享卡上离线补出片兜底。
- **8↔4 卡 resume 的 sharded state**：Stage 3 FSDP 分片 state 换卡数 resume 会校验 `cf_resume_meta.json`（`runner_dmd.py:229-258`）——无该文件走 legacy 分支（只 warn 不 block）。Stage 1/2 是 DDP 全量副本，无此检查，但要手动保证 `NUM_PROCESSES` 与训练动态一致。
- **Stage 3 EMA resume**：涉及 FSDP / world-size 相关的 `ema-rank{N}.safetensors` 与普通 `ema.safetensors` 两类路径，换卡数时要核对。
- **FSDP idle-grad**：修复已 baked-in `runner_dmd.py:63-74`；`scripts/fsdp_idle_grad_probe.py` 仅作历史复现，不在训练路径。
- **Stage 3 显存**：FSDP 兜底需 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（`launch/stage3_dmd.sh` 默认已设）；`mixed_precision: "no"` 是有意为之（FSDP 下禁 fp32 master，省 ~16GB/rank）。

## 4. 扩展点

- **换数据集**：改 `configs/*.yaml` 的 `metadata_path` / `dataset_base`（共享默认见 `default.yaml`），或改 `modules/dataset.py::build_cf_dataset`。
- **换 backbone**：改各 yaml 的 `student_init` / `base_ckpt`（teacher/critic 都是 5B 双向 base，见 `default.yaml:base_ckpt`）。
- **加 Stage 4（推理 / 评测）**：新增 `modules/<name>.py` runner + `train.py::main` 的 `--stage <name>` 分支（`train.py:49,84-88`）。可像 Stage 1/2 内联 `_run_*`，也可像 Stage 3 拆 `modules/runner_dmd.py`。
- **改 Stage 3 训练 regime**：长 / 短 rollout、score-frames、KV-cache window 全在 yaml 控制（对比 `stage3_dmd.yaml` 与 `stage3_dmd_long_from11200.yaml` 的 `dmd_long_rollout` / `dmd_score_frames` / `dmd_train_kv_window`）。

## 5. 目录结构

```
.
├── PLAN.md / implement.md / README.md / stage3_cf_alignment_review.md
├── train.py                       # 入口: --stage {ar_tf,cd,dmd} 派发
├── modules/
│   ├── ar_tf.py                   # Stage 1 training module
│   ├── cd.py                      # Stage 2 training module
│   ├── dmd.py                     # Stage 3 training module
│   ├── runner_dmd.py              # Stage 3 训练 runner (FSDP / EMA / resume)
│   ├── dataset.py                 # build_cf_dataset
│   └── ema.py                     # fp32 shadow EMA helpers (Stage 2/3 共享)
├── inference/
│   ├── sanity_sample.py           # KV-cache 自回归出片
│   ├── spawn_sample.py            # 非阻塞 subprocess 调度
│   ├── fixed_sanity.py            # fixed prompt + action
│   └── test_stage3_ckpts.py       # stage-agnostic 批量离线评测引擎
├── configs/
│   ├── default.yaml               # 共享键
│   ├── stage1_ar.yaml             # canonical (CF++ aligned, _2 lineage)
│   ├── stage2_cd.yaml             # canonical
│   ├── stage3_dmd.yaml            # canonical (26-window)
│   ├── stage3_dmd_long_from11200.yaml  # 长程 rollout 延伸 run
│   └── archive/                   # 旧 8card/4card + 一次性 ablation
├── launch/
│   ├── stage{1_ar,2_cd,3_dmd}.sh  # 主 launcher
│   ├── test_stage{1,3}.sh         # 离线批量 sanity
│   ├── watch_stage3.sh            # 训练健康 watcher
│   └── archive/                   # 一次性 / 硬编码 launcher
└── scripts/
    ├── verify.py                  # CPU+GPU 验证子命令 (state_dict/tf_mask/cd/rollout/dmd/...)
    └── fsdp_idle_grad_probe.py    # 历史复现, 不在训练路径
```

模型 / pipeline 在 `diffsynth/`（自包含，不引用 SF 项目同名文件）：

- `diffsynth/models/reactive_gwm_casual_forcing_dit.py`
- `diffsynth/pipelines/reactive_gwm_casual_forcing.py`
