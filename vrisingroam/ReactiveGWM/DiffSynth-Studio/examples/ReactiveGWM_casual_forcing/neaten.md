# ReactiveGWM Causal Forcing —— 三 Stage 完成后的整理计划

**Repo 根**：`/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/`

## Context

这是个 3-Stage 自回归蒸馏训练代码库（Stage 1 AR-TF、Stage 2 Causal Consistency Distillation、Stage 3 DMD，输入是 SF3 双向 Wan2.1-5B base，产出 4-step KV-cache 因果生成器）。所有 3 个 Stage 都已训练并验证通过。现状：
- **训练代码**：`train.py` Stage1/2 把 epoch loop 内联（各 ~150 行脚手架，与 Stage3 的 `runner_dmd.py` 大量同构）；`modules/` 混放训练 module（ar_tf/cd/dmd）和工具（ema/dataset）。
- **配置**：每 Stage 都有 2-3 个 yaml（canonical 4card + 8card legacy + 一次性 ablation 各 1 份），其中只有 `*_4card`（Stage 1/2）和 `stage3_dmd.yaml`（Stage 3）是真正在跑的。
- **启动脚本**：3 个主 launcher (`stage{1,2,3}_*.sh`) 共享大段样板；2 个 ad-hoc launcher（`resume_stage3_4card.sh` 硬编码 state-4800/GPU 0-3，`watch_stage1_aligned.sh` 硬编码 log 路径和 tmux session）一次性绑定到具体 run。
- **文档**：`README.md` 说 "Stage 1 进行中、Stage 2/3/4 Locked"（4 天前对，今天已经全部 done）；`PLAN.md`/`implement.md` 有具体可定位的 stale 句子；`stage3_cf_alignment_review.md` 是 point-in-time 审计、durable。

**目标（用户口径）**：更整洁、更方便复现、扩展性更强。已确认的范围决策：
1. **Moderate** 深度：归档+文档+抽 5 个 Python helper + 1 个 shell helper。**不**重命名目录、**不**把 Stage1/2 也拆 runner、**不**引入 base class。
2. 已废 yaml/shell 移到 `configs/archive/` 和 `launch/archive/`，不删。
3. README 面向自己 + 同组复现，保留绝对路径。
4. "扩展点" 只在 README 加一节，不写单独的 EXTENDING.md。

**Karpathy 约束**：每行新代码必须 1-to-1 对应一处已明确点名的重复。不为未来需求加抽象。

## 执行前补丁说明

- 本文件是整理 plan，不是代码变更记录；当前这轮只完善 `neaten.md` 内容，真正执行 Step 1-5 要另起执行步骤。
- Step 3 的文档刷新必须以执行时的 `grep` / `Read` 结果为准，不按旧行号机械修改。
- 已发现 `PLAN.md` 里有两处原本被标为 stale 的内容其实已经正确或已修正；见 Step 3.2 的“核对不强改”。

---

## 改动总览

5 步，**每步独立可 commit**。Step 1-3 是低风险硬通货；Step 4-5 是可跳过的代码 refactor，任何验证失败就 revert。

| 步骤 | 内容 | 风险 | 可跳过 |
|---|---|---|---|
| 1 | 归档 6 个已废 yaml/sh 到 `*/archive/` | 极低 | 否 |
| 2 | 2 个主 launcher 的 CFG 默认指向 canonical 4card 配置 | 低 | 否 |
| 3 | 重写 README，touch-up PLAN/implement | 极低 | 否 |
| 4 | 抽 `launch/_common.sh`（只共享 path/ARGS/env 设置） | 中（需 dry-run diff 字节级比对） | 是 |
| 5 | 抽 `modules/_common.py`（5 个 helper） | 中（需 verify.py + resume smoke 通过） | 是 |

---

## Step 1 — 归档

新建两个目录：
- `configs/archive/`
- `launch/archive/`

`git mv` 以下 6 文件（路径用 repo 相对）：

| 从 | 到 |
|---|---|
| `configs/stage1_ar.yaml` | `configs/archive/stage1_ar.yaml` |
| `configs/stage1_ar_aligned.yaml` | `configs/archive/stage1_ar_aligned.yaml` |
| `configs/stage2_cd.yaml` | `configs/archive/stage2_cd.yaml` |
| `configs/stage3_dmd_26win_resume5400.yaml` | `configs/archive/stage3_dmd_26win_resume5400.yaml` |
| `launch/resume_stage3_4card.sh` | `launch/archive/resume_stage3_4card.sh` |
| `launch/watch_stage1_aligned.sh` | `launch/archive/watch_stage1_aligned.sh` |

每个 archive 目录加一个简短 `README.md`（每个文件一行：来源 / 归档日期 2026-06-04 / 一句话原因）。

**留在顶层**：`configs/default.yaml`、`configs/stage1_ar_4card.yaml`、`configs/stage2_cd_4card.yaml`、`configs/stage3_dmd.yaml`、`launch/stage{1_ar,2_cd,3_dmd}.sh`、`launch/watch_stage3.sh`、`launch/test_stage{1,3}.sh`。

**验证**：`git status --porcelain` 只显示 6 个 rename + 2 个 archive README；`grep -rn "stage1_ar\.yaml\|stage1_ar_aligned\|stage2_cd\.yaml\|stage3_dmd_26win\|resume_stage3_4card\|watch_stage1_aligned" configs/ launch/ inference/ scripts/ modules/ train.py` 运行路径里不应命中。archive README / 历史文档里出现旧文件名可以接受。

---

## Step 2 — Launcher 默认指向 canonical 配置

只改两行：

- `launch/stage1_ar.sh:19` —— `CFG="${CFG:-${HERE}/configs/stage1_ar.yaml}"` → `CFG="${CFG:-${HERE}/configs/stage1_ar_4card.yaml}"`
- `launch/stage2_cd.sh:25` —— 同理改为 `stage2_cd_4card.yaml`

`launch/stage3_dmd.sh:29` 不动（已经指向 `stage3_dmd.yaml`）。

**验证**：`bash -nx launch/stage1_ar.sh 2>&1 | head` 显示 CFG 路径已切换，其他 trace 行字节相同。

---

## Step 3 — 文档刷新

### 3.1 `README.md` 整体重写

按下面 6 节的骨架重写(旧的 75 行 status 表全部丢弃):

1. **概览**(~10 行) —— 1 句话系统描述 + "All 3 stages trained and validated as of 2026-06-04" + 指向 `PLAN.md` / `implement.md` / `stage3_cf_alignment_review.md`。
2. **最终产物 & 路径**(~15 行) —— 每个 stage 的 final `step-N.safetensors` 绝对路径，必须从对应 canonical yaml 的 `output_path` + 实际目录里最大 step 核对，不手写猜测。Stage 3 还包括 `stage3_dmd_generator.safetensors`、`stage3_dmd_ema.safetensors`、`stage3_dmd_critic.safetensors` 三件套(见 `runner_dmd.py:438/446/454`)。
3. **复现 Recipe**(~25 行) —— 每 stage 一段:`CUDA_VISIBLE_DEVICES=... NUM_PROCESSES=... bash launch/stage{1,2,3}_*.sh`,下一 stage 的 `STUDENT_INIT` 指向上一 stage 的产物;包含 `RESUME_STATE=.../state-N bash ...` 形式;包含 `bash launch/test_stage{1,3}.sh` 离线 sanity。
4. **已知陷阱**(~15 行 bullets) —— (a) 训练中 sanity OOM 用 `test_stage*.sh` 兜底;(b) 8↔4 card resume 的 sharded state 兼容(Stage 3 的 `cf_resume_meta.json` 检查见 `runner_dmd.py:229-258`,Stage 1/2 无该检查,需手动同步 NUM_PROCESSES);(c) Stage 3 EMA resume 同时涉及 FSDP/world-size 相关的 `ema-rank{N}.safetensors` 与普通 `ema.safetensors` 路径，换卡数时要核对；(d) FSDP idle-grad 修复已 baked-in `runner_dmd.py:63-74`,`scripts/fsdp_idle_grad_probe.py` 仅作历史复现;(e) Stage3 FSDP 兜底要 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(`stage3_dmd.sh:42` 默认设了)。
5. **扩展点**(~12 行 bullets) —— (a) 换数据集:改 `modules/dataset.py::build_cf_dataset` 或者只换 cfg 的 `metadata_path/dataset_base`(Step 5 后所有 stage 通过同一 `build_dataloader`);(b) 换 backbone:改 `student_init`/`teacher_ckpt`/`base_ckpt`,注意 teacher/critic 都是 5B;(c) 加 Stage 4(推理/评测):新加 `modules/<name>.py` runner + `train.py::main` 的 `--stage <name>` 分支;既可像 Stage1/2 内联 `_run_*`、也可像 Stage 3 拆 `modules/runner_*.py`,5 个 `_common.py` helper 已覆盖样板。
6. **目录索引**(~8 行 table) —— `configs/{,archive/}`、`launch/{,archive/,_common.sh}`、`modules/{ar_tf,cd,dmd,runner_dmd,dataset,ema,_common}.py`、`inference/`、`scripts/verify.py`；保留 `diffsynth/models/reactive_gwm_casual_forcing_dit.py` 与 `diffsynth/pipelines/reactive_gwm_casual_forcing.py` 两个实际模型 / pipeline 入口说明。

### 3.2 `PLAN.md` 微调

只动仍然需要动的内容；对已正确内容只核对不强改：

- **顶部加 1 行**："Status (2026-06-04): all 3 stages trained and validated. See implement.md for run-by-run status and README.md for reproduction."
- **§ 7 teacher/critic 大小核对**：`grep -n "14B" PLAN.md` 定位。当前 `PLAN.md:136` 的 `14B` 文字已经是正确解释：论文用 14B score model，但本项目只有 5B 双向 base，teacher/critic 都是 5B。这里**保留当前说明，不改成纯 5B 句子**。
- **§ 2 Stage 3 文件引用核对**：当前 `PLAN.md:132` 已经修正原先不存在的 `pipeline/causal_forcing_training.py` / `utils/dmd_loss.py` 引用，并指向 `stage3_cf_alignment_review.md`。这里**只核对，不重复编辑**。

其余不动（durable design 内容）。

### 3.3 `implement.md` 微调

不要只用 `grep -n "Stage 1\|Stage 2\|Stage 3" implement.md` 找 3 处状态块；实际 stale 分散。至少核对并更新：

- `implement.md:3` 顶部总状态块：Stage 1 暂停、Stage 2 进行中等当前态。
- `implement.md:46-49` 总览表：Stage 1 暂停、Stage 2 进行中、Stage 3 pending、Stage 4 Locked。
- `implement.md:182-183` Stage 1 full run 仍进行中 / step 14060。
- `implement.md:249-255` Stage 1 当前状态仍是 2026-05-21 暂停态。
- `implement.md:259-262` Stage 2 标题和状态仍是 50K run 进行中。

把这些当前状态都改为完成态条目（最终 step + 日期 + final ckpt 路径），每条带 "Updated 2026-06-04" 标记。

不动：历史 progress 条目、design rationale、decisions log。旧 step（如 14060 / 12500 / 17500）允许保留在历史 log，但不应出现在“当前状态”总结里。

### 3.4 `stage3_cf_alignment_review.md`

**不动**。point-in-time 审计已完成。

**验证 Step 3**：
- `grep -n "Locked\|🔒" README.md` 应无命中。
- `grep -n "14B" PLAN.md` 只允许在“论文 14B vs 本项目 5B”的正确说明语境内命中。
- `grep -n "50K run active\|verify/dry-run pending\|Locked\|🔒\|进行中\|pending\|active" implement.md` 不应在当前状态总结中命中。
- `grep -n "14060\|12500\|17500" implement.md` 只允许出现在历史 log / decisions，不应出现在当前状态总结中。

---

## Step 4 —（可选）`launch/_common.sh` 抽取

### 文件：`launch/_common.sh`（新建）

**只共享**这些（path/env setup + ARGS 构建）：

```bash
#!/usr/bin/env bash
# 三 stage 主 launcher 共享: 路径/环境/--stage --config --resume_state 三个 flag。
# 不抽 accelerate launch 本身 —— 三 stage 调用形态差异大（Stage 3 有 LAUNCH_FLAGS 数组 +
# FSDP 分支），抽进来会把 Stage3 特殊性挤进通用脚本。

set -euo pipefail

# 用 BASH_SOURCE[1] = 调用者 (各 stage launcher), 不是 _common.sh 本身。
HERE="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
TRAIN_PY="${HERE}/train.py"

cf_build_args() {
    # 用法: ARGS=(); cf_build_args <stage> <cfg_path>
    # 把 --stage X --config Y [--resume_state Z] 推入调用者作用域的 ARGS 数组。
    ARGS+=(--stage "$1" --config "$2")
    if [[ -n "${RESUME_STATE:-}" ]]; then
        ARGS+=(--resume_state "${RESUME_STATE}")
    fi
}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
```

### 三个主 launcher 改写

每个 launcher 顶部 `source` `_common.sh`，自己写 `accelerate launch` 调用（保留 stage-specific 部分）：

**`launch/stage1_ar.sh`**（精简到 ~15 行）：
```bash
#!/usr/bin/env bash
# Stage 1 AR-TF launch — accelerate DDP（OOM 切 FSDP 单 unit, 用 accelerate config）。
# Env: BASE_CKPT/DATA_ROOT/OUT/STUDENT_INIT/RESUME_STATE/NUM_PROCESSES/MAIN_PORT
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CFG="${CFG:-${HERE}/configs/stage1_ar_4card.yaml}"
ARGS=()
cf_build_args ar_tf "${CFG}"

accelerate launch \
    --num_processes "${NUM_PROCESSES:-8}" \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PORT:-29520}" \
    "${TRAIN_PY}" "${ARGS[@]}"
```

**`launch/stage2_cd.sh`**：同形态，CFG 默认 `stage2_cd_4card.yaml`，端口 29521。

**`launch/stage3_dmd.sh`**：保留 `LAUNCH_FLAGS` 数组 + FSDP 分支 + `PYTORCH_CUDA_ALLOC_CONF` 导出 + 末尾 `accelerate launch "${LAUNCH_FLAGS[@]}" "${TRAIN_PY}" "${ARGS[@]}"`，只把开头的 `set/HERE/REPO_ROOT/TRAIN_PY/ARGS=/RESUME_STATE/cd/PYTHONPATH/TOKENIZERS_PARALLELISM/PYTHONUNBUFFERED` 换成 `source _common.sh + cf_build_args dmd "${CFG}"`。

### 不进 `_common.sh` 的内容（明确排除）

- `PYTORCH_CUDA_ALLOC_CONF`（Stage3-only）
- FSDP 分支（Stage3-only）
- `accelerate launch` 本身 + `--multi_gpu / --mixed_precision / --main_process_port` 默认值（三 stage 各异）
- `MIXED_PRECISION` 默认（Stage 1/2 用 `bf16`，Stage 3 用 `no`）

### 验证（关键 gate）

对三个 launcher 各做一次 dry-run trace 比对：
```bash
bash -nx launch/stage1_ar.sh 2>&1 | grep -v '^+ source\|^+ HERE=\|^+ REPO_ROOT=\|^+ TRAIN_PY='
# 与 Step 4 前的同命令输出做 diff, 必须只在 CFG 路径上不同（Step 2 已经改过的）。
```

`accelerate launch ... train.py --stage X --config ...` 行必须**逐字符相同**（CFG 路径在 Step 2 已切换）。额外核对：
- `PYTHONPATH`、`TOKENIZERS_PARALLELISM`、`PYTHONUNBUFFERED` 的值不变。
- `RESUME_STATE` 仍只在非空时加入 `ARGS`。
- Stage 3 的 `PYTORCH_CUDA_ALLOC_CONF`、FSDP 分支、`LAUNCH_FLAGS` 数组不进入 `_common.sh`，且 dry-run trace 不变。

---

## Step 5 —（可选）`modules/_common.py` 抽取

### 文件：`modules/_common.py`（新建）

5 个 helper，每个都 1-to-1 对应一处已点名的重复。文件头 docstring 列出 5 个 helper 名 + 各自吃掉的重复位置（train.py / runner_dmd.py 行号）。

#### Helper 1：`build_dataloader(cfg, num_workers=None) -> torch.utils.data.DataLoader`

吃掉：
- `train.py:130-140` + `167-172`（Stage 1）
- `train.py:286-296` + `336-341`（Stage 2）
- `runner_dmd.py:101-111` + `135-137`（Stage 3）

内部：调 `build_cf_dataset(...)` 读 9 个 cfg key（与 `train.py:130-140` 完全相同的默认值），包到 `DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers or int(cfg.get("dataset_num_workers", 2)))`。

#### Helper 2：`build_accelerator(cfg, *, find_unused_parameters=False, gradient_accumulation_steps=None) -> accelerate.Accelerator`

吃掉：
- `train.py:121-127`（Stage 1，默认参数）
- `train.py:278-284`（Stage 2，默认参数）
- `runner_dmd.py:92-98`（Stage 3，`find_unused_parameters=True, gradient_accumulation_steps=1`）

内部：`gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1)) if gradient_accumulation_steps is None else gradient_accumulation_steps`；`mixed_precision = cfg.get("mixed_precision", "bf16")`；DDP kwargs handler 用入参 `find_unused_parameters`。

#### Helper 3：`extract_common_hparams(cfg) -> dict[str, Any]`

吃掉：
- `train.py:151-158`（Stage 1，8 项全用）
- `train.py:319-326`（Stage 2，8 项全用；`ema_start_step` 在 `327` 仍保留 inline）
- `runner_dmd.py:119, 121, 126, 128, 129, 130`（Stage 3，6 项；其余 Stage3-only 留 inline）

返回 dict，key 名（**不带前缀**）：`{"lr", "weight_decay", "beta1", "beta2", "num_workers", "save_steps", "max_steps", "grad_clip"}`。读取的 cfg key 仍是带前缀的 `ar_beta1/ar_beta2/ar_grad_clip/dataset_num_workers`，默认值与现状逐一对齐（`lr=2.0e-6, weight_decay=0.01, beta1=0.0, beta2=0.999, num_workers=2, save_steps=2500, max_steps=50000, grad_clip=10.0`）。

⚠️ **Stage 3 的 key 差异**（**load-bearing，必须保留**）：runner_dmd.py:122-123 读的是 `cfg.get("beta1", 0.0)` 和 `cfg.get("beta2", 0.999)`，**不带** `ar_` 前缀。Step 5 在 Stage 3 调完 `extract_common_hparams` 之后必须显式覆盖：
```python
_h = extract_common_hparams(cfg)
beta1 = float(cfg.get("beta1", 0.0))    # Stage 3 cfg 用裸 beta1（非 ar_beta1）, 保留
beta2 = float(cfg.get("beta2", 0.999))
# 其他从 _h 取
```
helper 的 docstring 明确写这一条，避免误用。

#### Helper 4：`parse_resume_step(resume_state_path: str) -> int`

吃掉：
- `train.py:188-190`（Stage 1）
- `train.py:379-381`（Stage 2）
- `runner_dmd.py:294-296`（Stage 3）

内部：`m = re.search(r"state-(\d+)/?$", resume_state_path.rstrip("/"))`；返回 `int(m.group(1))` 或 `0`。

#### Helper 5：`summon_full_params_ctx(model)` —— context manager

吃掉：
- `train.py:349-364`（Stage 2 `_full_params_for_export`）
- `runner_dmd.py:189-204`（Stage 3 `_full_params_for_export`）

内部：两处逻辑相同，但 docstring 不完全相同；合并时取更完整的说明，不改变行为（try-import FSDP；`isinstance(model, FSDP)` 时返回 `FSDP.summon_full_params(model, recurse=True, writeback=False, rank0_only=True, offload_to_cpu=True)`；否则 `contextlib.nullcontext()`）。

### 三个 runner 的修改 blueprint

**train.py `_run_ar_tf` (Stage 1)**：
- import 加 `from examples.ReactiveGWM_casual_forcing.modules._common import build_accelerator, build_dataloader, extract_common_hparams, parse_resume_step`。
- 替换 `121-127` → `accelerator = build_accelerator(cfg)`。
- 删 `130-140` 的 dataset 块（合并进 dataloader）。
- 替换 `151-158` → `_h = extract_common_hparams(cfg)` + 8 个本地名绑定。
- 替换 `167-172` → `dataloader = build_dataloader(cfg, num_workers=num_workers)`。
- 替换 `187-190` → `step = parse_resume_step(args.resume_state)`。
- 其余（`ModelLogger`、AdamW、`accelerator.prepare`、resume `accelerator.load_state` + `logger.num_steps = step` 同步、主循环、save+sanity spawn）**逐字保留**。

**train.py `_run_cd` (Stage 2)**：
- import 加 `..._common import build_accelerator, build_dataloader, extract_common_hparams, parse_resume_step, summon_full_params_ctx`。
- 替换 `278-284` → `accelerator = build_accelerator(cfg)`。
- 删 `286-296` 的 dataset 块。
- 替换 `319-326` → `_h = ...` + 8 个本地名。**`327` 的 `ema_start_step` 保留 inline**。
- 替换 `336-341` → `dataloader = build_dataloader(cfg, num_workers=num_workers)`。
- 删 `_full_params_for_export` 定义（`349-364`），把 `_save_cd_export`（`366-371`）里的 `with _full_params_for_export():` 改为 `with summon_full_params_ctx(model):`。
- 替换 `379-381` → `step = parse_resume_step(args.resume_state)`。
- 其余（Stage 2 settings 打印、CFCDTrainingModule、EMA 手动 resume、主循环、`update_ema()`、save+ema.safetensors、final-tail save）**逐字保留**。

**modules/runner_dmd.py `run_dmd` (Stage 3)**：
- import 加 `..._common import build_accelerator, build_dataloader, extract_common_hparams, parse_resume_step, summon_full_params_ctx`。
- 替换 `92-98` → `accelerator = build_accelerator(cfg, find_unused_parameters=True, gradient_accumulation_steps=1)`。
- 删 `101-111` 的 dataset 块。
- 替换 `119, 121, 126, 128, 129, 130` 的 6 行共享 hparam 抽取 → `_h = extract_common_hparams(cfg)` + 6 个本地名；**Stage 3 专有的 `lr_critic/beta1_critic/beta2_critic/ratio/ema_start_step/empty_cache_each_phase/gc_interval` 保留 inline**；**`beta1/beta2` 必须在 helper 调用后用裸 key 显式覆盖**（见 Helper 3 警告）。
- 替换 `135-137` → `dataloader = build_dataloader(cfg, num_workers=num_workers)`。
- 删 `_full_params_for_export` 定义（`189-204`），把 `_save`、critic save、final-products 三处 `with _full_params_for_export():` 改为 `with summon_full_params_ctx(model):`。
- 替换 `294-296` → `step = parse_resume_step(args.resume_state)`。
- 其余（`CFDMDTrainingModule`、双 param-group AdamW、`prepare(model)` + `prepare(optimizer, scheduler, dataloader)` 的两次 prepare 顺序——**FSDP load-bearing，绝不改**、`_mem` 插桩、`_is_fsdp_wrapped`、`_world_size`、`_resume_meta_*`、`_ema_rank_path/_save_ema_state/_load_ema_state`、resume validate、主交替循环、`_none_out_grads`、save、final products）**逐字保留**。

### `modules/__init__.py`

**不动**。helper 用全路径 `examples.ReactiveGWM_casual_forcing.modules._common` 导入，对齐 `train.py:111-113` 注释说明的 path-collision 规避策略（不要在 `__init__.py` re-export）。

---

## 关键文件清单

会被本次整理触碰的文件：

- `/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/train.py`（Step 5）
- `/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/modules/runner_dmd.py`（Step 5）
- `/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/modules/_common.py`（新建，Step 5）
- `/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/launch/_common.sh`（新建，Step 4）
- `/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/launch/stage{1_ar,2_cd,3_dmd}.sh`（Step 2、Step 4）
- `/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/README.md`（Step 3，重写）
- `/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/PLAN.md`（Step 3，行级 touch-up）
- `/nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/implement.md`（Step 3，行级 touch-up）
- 6 个归档移动的文件（Step 1）

复用的现有工具：
- `modules/ema.py`（3 个 fp32 shadow EMA helper）—— 已在 Stage 2/3 间共享，**不动**。
- `modules/dataset.py::build_cf_dataset`—— 仍在 `_common.build_dataloader` 内部调用。
- `inference/{sanity_sample,spawn_sample,fixed_sanity,test_stage3_ckpts}.py`、`scripts/verify.py`、`scripts/fsdp_idle_grad_probe.py`—— 全部**不动**。
- `diffsynth/pipelines/reactive_gwm_casual_forcing.py`—— 上游算法库，本次不跨 example/diffsynth 边界改动。

---

## 验证（每步独立执行）

### Step 1（归档）
```bash
git status --porcelain   # 只看到 6 rename + 2 archive README
grep -rn "stage1_ar\.yaml\b\|stage1_ar_aligned\b\|stage2_cd\.yaml\b\|stage3_dmd_26win\|resume_stage3_4card\|watch_stage1_aligned" \
    configs/ launch/ inference/ scripts/ modules/ train.py
# 运行路径里不应命中；archive README / 历史文档里的旧文件名可以接受
```

### Step 2（CFG 默认）
```bash
bash -nx launch/stage1_ar.sh 2>&1 | tail -10   # CFG 路径切到 _4card
bash -nx launch/stage2_cd.sh 2>&1 | tail -10   # 同上
# 与 Step 2 前对比, 仅 CFG 字符串不同
```

### Step 3（文档）
```bash
grep -n "Locked\|🔒" README.md          # 无命中
grep -n "14B" PLAN.md                    # 只允许在“论文 14B vs 本项目 5B”的正确说明语境内
grep -n "50K run active\|verify/dry-run pending\|Locked\|🔒\|进行中\|pending\|active" implement.md
grep -n "14060\|12500\|17500" implement.md
# 后两条不应在当前状态总结中命中；旧 step 只允许出现在历史 log / decisions
```

### Step 4（`_common.sh`）—— gate
```bash
# 每个 launcher 跑 dry-run trace, 与 Step 4 前 diff
for f in launch/stage1_ar.sh launch/stage2_cd.sh launch/stage3_dmd.sh; do
    bash -nx "$f" 2>&1 > /tmp/post_${f##*/}.trace
    diff /tmp/pre_${f##*/}.trace /tmp/post_${f##*/}.trace
    # 期望: 仅 source/HERE/REPO_ROOT 行的位置变化, accelerate launch 行字节相同
    # 同时核对 PYTHONPATH/TOKENIZERS_PARALLELISM/PYTHONUNBUFFERED、RESUME_STATE 条件加入 ARGS、Stage3 LAUNCH_FLAGS/FSDP/PYTORCH_CUDA_ALLOC_CONF 均不变
done
```

### Step 5（`_common.py`）—— gate

5 项**都**必须通过：

1. **Static**：
   ```bash
   python -m py_compile examples/ReactiveGWM_casual_forcing/train.py \
       examples/ReactiveGWM_casual_forcing/modules/_common.py \
       examples/ReactiveGWM_casual_forcing/modules/runner_dmd.py
   PYTHONPATH=$(pwd) python -c "from examples.ReactiveGWM_casual_forcing.modules._common \
       import build_dataloader, build_accelerator, extract_common_hparams, parse_resume_step, \
              summon_full_params_ctx; print('ok')"
   ```

2. **CPU verify**（必跑）：
   ```bash
   python examples/ReactiveGWM_casual_forcing/scripts/verify.py state_dict
   python examples/ReactiveGWM_casual_forcing/scripts/verify.py tf_mask
   python examples/ReactiveGWM_casual_forcing/scripts/verify.py kv_cache_rollout
   ```
   3 个 subcommand 都必须像 refactor 前一样过。

3. **GPU verify**（如有空闲 GPU）：
   ```bash
   python examples/ReactiveGWM_casual_forcing/scripts/verify.py cd
   python examples/ReactiveGWM_casual_forcing/scripts/verify.py rollout
   python examples/ReactiveGWM_casual_forcing/scripts/verify.py dmd
   ```

4. **Resume smoke**（per stage）：
   每 stage 准备临时 yaml，把 `max_steps` 设成已 resume step 的值（比如已知 state-N 的 N），让循环 break 立即退出：
   ```bash
   # Stage 1 例:
   cp configs/stage1_ar_4card.yaml /tmp/cf_smoke_s1.yaml
   echo "max_steps: <已知state-N的N>" >> /tmp/cf_smoke_s1.yaml
   NUM_PROCESSES=1 CFG=/tmp/cf_smoke_s1.yaml \
       RESUME_STATE=<已知state-N目录> \
       bash launch/stage1_ar.sh
   # 期望: 打印 "[Resume] loaded state from ... (step=N)" 后 1 个 micro-batch 内退出, 无 traceback
   ```
   Stage 2、Stage 3 同样做一遍（Stage 3 注意 `cf_resume_meta.json` 拓扑要匹配 NUM_PROCESSES）。

5. **Dry-run trace 不退化**：Step 4 已经过的 launcher dry-run diff，加 Step 5 后再过一次，确认 `train.py --stage X --config Y` 调用不变。

**任一项失败 → revert Step 5，停在 Step 4 已经是干净状态**。

---

## 显式不做（防止执行漂移）

- 不重命名 `modules/` → `trainers/`。
- 不把 Stage1/2 也拆 `runner_*.py`。
- 不加 `BaseTrainer/BaseRunner/mixin/registry/decorator`。
- 不加 wandb/mlflow/tensorboard/新 metric 流。
- 不加 `train.py` 或 launcher 新 CLI flag。
- 不动 `inference/`、`scripts/verify.py`、`scripts/fsdp_idle_grad_probe.py`。
- 不删任何文件（已废的归档不删）。
- 不动 `configs/default.yaml` 和 3 个 canonical yaml 内容。
- 5 个 Python helper + `_common.sh` 里的 1 个函数 + 4 行 env 导出 之外，不引入任何新抽象。
