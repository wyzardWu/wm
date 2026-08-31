# ReactiveGWM_casual_forcing — 实现进度 (implement.md)

> 状态 (Updated 2026-06-06)：**Stage 1/2/3 均已训练并验证通过**（`sf3_casual_forcing_2` lineage）。Stage 1 CF++ aligned 重训到 step-24000；Stage 2 CD 到 step-20000；Stage 3 DMD 26-window 到 step-11400，长程 rollout 延伸 run (`stage3_dmd_long_from11200`) 进行中。各 stage canonical 配置见 `configs/stage{1_ar,2_cd,3_dmd}.yaml` 与 `configs/stage3_dmd_long_from11200.yaml`；复现见 [README.md](./README.md)。下方历史 progress 条目保留原状（含旧谱系 step / 决策日志），仅顶部总状态更新为完成态。
> 创建日期：2026-05-20。
> 配套设计：[PLAN.md](./PLAN.md)。
> 更新规范：按 [`mark-progress-in-implement-md`](../../memory/mark-progress-in-implement-md.md)，每个子任务完成后即时把 `[ ]` 改成 `[x]` 并在底部"变更日志"追加一行。

---

## 工作原则

1. **按 Stage 分批实现**。Stage N 完成、acceptance 全过、用户 review 通过后，**才**进入 Stage N+1。
2. **代码增量式**。Stage 1 写 Stage 1 必需的代码（包含 KV-cache，与 CF 上游对齐）；Stage 2 在 Stage 1 代码基础上**新增**（不重写已有）；Stage 3 同理（在 Stage 1/2 基础上新增 DMD，不重写 model/pipeline 的 KV-cache）。
3. **Acceptance 包含"训练出符合预期的 ckpt"**——不只是 verify 单测通过，要在 8×H200 上跑出可用产物 + loss 曲线合理 + 中途的 sanity 推理视频肉眼合理。
4. **不动 SF 项目**。`examples/ReactiveGWM_self_forcing/` 与 `diffsynth/{models,pipelines}/reactive_gwm_self_forcing*` 全程不读、不引用、不复用。
5. **Review Gate 不可跨越**。用户没说"OK 进 Stage X"前，永远不要开始 Stage X 的代码或文件。
6. **每个 Stage 的 verify 与 acceptance 项都列在本文件**——勾完才算 Stage 完成。

---

## 通用训练规格（Stage 1 / 2 / 3 共用，可在各 stage yaml 覆盖）

| 项 | 值 |
|---|---|
| 数据 | `/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF3/metadata.csv` **全量**（不抽样） |
| 训练视频窗口 | SF3 profile，`num_frames=101` 像素帧（100 帧以内最大化）→ `(101-1)//4+1 = 26` latent 帧 |
| 训练步数 | **`max_steps=50000`** |
| save 节奏 | **每 2500 step** 保存完整 checkpoint（model + optimizer + LR scheduler + RNG + step + accelerate state），**任一 ckpt 可完整 resume** |
| sanity 推理 | **每 2500 step**（与 save 同节奏）由主进程 spawn **非阻塞 subprocess** 推理一段 sanity 视频；**绝不阻塞或干扰训练 main loop** |
| sanity 推理路径 | **所有 Stage 统一走 KV-cache + block-by-block 自回归**（CF 上游 `pipeline/causal_diffusion_inference.py` 风格）；与训练用的 TF dual-block 路径互补 |
| sanity 视频长度 | **60 秒 ≈ 1500 像素帧 ≈ 300 latent 帧 @ 25 FPS**（KV-cache 自回归，长度任意，三阶段都能做到） |
| sanity 单帧扩散步数 | Stage 1 / 2：**50 步** multi-step flow-match per frame；Stage 3：**4 步**（`denoising_step_list=[1000,750,500,250]`） |
| sanity 推理 GPU 策略 | 默认 `CUDA_VISIBLE_DEVICES=0` + `torch.cuda.set_per_process_memory_fraction(0.2)`（与训练共享 GPU 0，硬限 ~16 GB；5B bf16 模型 + KV-cache ≈12 GB 可入）；fallback `--device cpu` bf16 |
| sanity 重叠保护 | 若上次 sanity 推理 subprocess 仍 alive 而下一保存点又到，**跳过本次推理**（log warning，训练不阻塞） |
| 8×H200 后端 | DDP 优先；OOM 切 FSDP 单 unit |

> **关于 KV-cache 提前到 Stage 1**：CF 上游 Stage 1 训练完后做长视频推理时，本来就走 `pipeline/causal_diffusion_inference.py` 的 KV-cache 自回归路径（一帧一帧 multi-step 去噪、K/V 累加 + sink+recent 驱逐）。我们对齐 CF 上游，把 KV-cache 路径在 `CausalForcingReactiveGWMModel` 里从 Stage 1 起就实现完整。Stage 3 的 DMD self-rollout 训练直接复用这套 KV-cache 路径，不再"追加 KV-cache 实现"。这把工作从 Stage 3 挪到 Stage 1，但 net code 量持平；好处是 Stage 1/2/3 三阶段 sanity 推理 helper 完全统一，且符合 CF 上游做法。
> Action 序列由 dataset clip 0 提供（≤101 像素帧）；超出 clip 长度的部分（102–1500 像素帧）用 **idle action（10 维全零按键）** 填充。

---

## 总览

| Stage | 名称 | 产物 | 状态 |
|---|---|---|---|
| 1 | AR Diffusion Teacher Forcing | `stage1_ar/step-24000.safetensors` | ✅ 完成（Updated 2026-06-06；CF++ aligned 重训, `sf3_casual_forcing_2/stage1_ar`，最终 step-24000） |
| 2 | Causal Consistency Distillation | `stage2_cd/step-20000.safetensors` (EMA 优先) | ✅ 完成（Updated 2026-06-06；`sf3_casual_forcing_2/stage2_cd`，student_init/teacher=stage1 step-24000，最终 step-20000；verify 11/11 全过） |
| 3 | Asymmetric DMD self-forcing | `stage3_dmd/step-11400` + `stage3_dmd_long_from11200/step-N`（EMA 权重） | ✅ 训练+验证通过（Updated 2026-06-06；26-window run 到 step-11400；长程 rollout 延伸 run `stage3_dmd_long_from11200` 进行中） |
| 4 | 推理与评测 | 4-step 视频 + 13 动作轴报告 | 🔒 Locked（未排期） |

---

## Stage 1 — AR Diffusion Teacher Forcing

### 1.0 目标

训练出 **`stage1_ar.safetensors`**：从双向 `ReactiveGWM_base` 出发，切到因果 + dual-block TF 注意力，用 multi-step flow-matching 训练得到 AR teacher，作为 Stage 2 CD 的 teacher 与 student 初始化。

**本 Stage 同时实现 `CausalForcingReactiveGWMModel` 的完整功能**（TF dual-block 训练路径 + KV-cache 自回归推理路径），为 Stage 2/3 复用打底。

训练规格按上方 [通用训练规格](#通用训练规格stage--1--2--3-共用可在各-stage-yaml-覆盖) 表执行。

### 1.1 范围（本 Stage 新增/修改的工件）

**`diffsynth/` 新增**（自包含、与 SF 文件无引用关系）：

- [x] `diffsynth/models/reactive_gwm_casual_forcing_dit.py`
  - `CausalForcingReactiveGWMModel(ReactiveGWMModel)`
  - **本 Stage 必须实现的完整路径**（移植自 CF 上游 `wan/modules/causal_model.py`）：
    - causal block-attention（block 内 dense、块间 causal）
    - dual-block TF forward（`clean_x` / `noisy_x` 双块 + `_prepare_teacher_forcing_mask` + flex_attention block mask + 四元组缓存）
    - **per-block KV-cache**（`init_kv_cache` / `update_kv_cache` / `evict_old_blocks`）：
      - sink+recent eviction（前 sink_size=2 个 block 永驻 + 最近 `kv_window_size=16` 个 block 滚动）
      - `forward(noisy_x, timestep, ..., kv_cache=...)` 走 KV-cache 路径：从 cache 读 K/V，新 K/V 写回
    - **cache-only refill 路径**：`refill_kv_cache(x, t=0, action, kv_cache)`——只算 K/V 写入 cache，不算 attention 输出（用于首帧 anchor 与每帧 rollout 完成后写入历史）
    - cross-attn cache（prompt 一次性写入、所有帧复用）
    - `seperated_timestep=True` 逐帧 timestep modulation
    - state dict 与双向 base 100% 兼容（不新增 `nn.Parameter`）
    - forward 签名 dispatch（按入参分发）：
      - `clean_x is not None` → TF dual-block（训练用）
      - `kv_cache is not None` → KV-cache 自回归（sanity 推理 / Stage 3 rollout / 最终推理用）
      - 都为 None → plain block-causal（调试 / verify 用）
  - **本 Stage 不写**：Stage 3 才用的 DMD 相关数据流（这部分在 module 层，model 层 forward 路径已经齐全）

- [x] `diffsynth/pipelines/reactive_gwm_casual_forcing.py`
  - `FlowMatchScheduler` wrapper（`shift=5.0`, `sigma_min=0.0`, 1000-step 离散）+ multi-step 采样 helper
  - `model_fn_causal_forcing(model, ..., tf_mode: bool, clean_x=None, noisy_x, timestep, kv_cache=None, ...)`：
    - `tf_mode=True, clean_x=x0`：TF dual-block 路径（训练用，Stage 1/2）
    - `tf_mode=False, kv_cache=<dict>`：KV-cache 自回归路径（sanity 推理 / Stage 3 rollout 用）
  - **本 Stage 不写**：CD step helper（Stage 2 加）；holistic DMD helpers（Stage 3 加）

**`examples/ReactiveGWM_casual_forcing/` 新增**：

- [x] `__init__.py`
- [x] `train.py`：argparse + yaml merge + accelerate + `--stage ar_tf` 派发；**含完整 checkpoint dump/resume**（model / optimizer / LR scheduler / Python+numpy+torch+cuda RNG / step / accelerate state）；**含 sanity 推理后台 spawn 集成**
- [x] `modules/__init__.py`
- [x] `modules/dataset.py`：SF3 26-latent-frame 窗口（GT latent + 动作序列 + prompt 字符串）；复用 `examples/ReactiveGWM/data/`（profile / action_utils / prompt_utils）
- [x] `modules/ar_tf.py`：`CFARTrainingModule(DiffusionTrainingModule)`，`forward(data) -> scalar_loss` 返回 TF dual-block flow-match MSE
- [x] `inference/__init__.py`
- [x] `inference/sanity_sample.py`：**KV-cache 自回归推理**（详见 1.2.10）；Stage 1/2 用 50 步、Stage 3 用 4 步——由 `--diffusion_steps` 参数控制
- [x] `inference/spawn_sample.py`：subprocess 非阻塞调度（`Popen` + 重叠保护 + GPU 共享 / CPU fallback）
- [x] `inference/fixed_sanity.py`：sanity 用的 fixed prompt + fixed action 序列加载器（从 dataset 第 0 个 clip 取 prompt + action；**action 不足 1500 像素帧时用 idle action（全零 10 维按键）填充到 1500**）
- [x] `configs/default.yaml`（所有共享键，对应 PLAN §7）
- [x] `configs/stage1_ar.yaml`（覆盖 `stage: ar_tf`、`max_steps: 50000`、`save_steps: 2500`、`sanity_sample_steps: 2500`、`sanity_sample_pixel_frames: 1500`、`sanity_sample_latent_frames: 300`、`sanity_sample_diffusion_steps: 50`、`sanity_sample_kv_window_size: 16`、`sanity_sample_sink_size: 2`、`sanity_sample_device: gpu_shared`、`sanity_sample_clip_idx: 0`、`output_path`）
- [x] `launch/stage1_ar.sh`：`accelerate launch train.py --stage ar_tf --config configs/stage1_ar.yaml`，8×H200
- [x] `scripts/verify.py`：CPU 子命令 `state_dict` / `scheduler` / `tf_mask` / `ar_tf` / `kv_cache_init` / `kv_cache_refill` / `kv_cache_rollout` / `sanity_kv` / `smoke`；GPU 子命令 `bf16_two_paths`
- [x] `README.md`：Stage 1 操作说明（环境、verify 跑法、launch 指令、ckpt 路径约定、resume 命令、sanity 视频路径）

> Stage 1 暂不创建：`modules/cd.py` / `modules/ema.py` / `modules/dmd.py` / `modules/runner_dmd.py` / `inference/causal_inference.py` / `inference/eval_action.py` / `configs/stage2_cd.yaml` / `configs/stage3_dmd.yaml` / `launch/stage2_cd.sh` / `launch/stage3_dmd.sh` / `launch/infer.sh`。

### 1.2 实现子步骤

- [x] **1.2.1** 骨架占位：所有 1.1 列出的文件创建为空模块或最小占位（仅 import + class/def 声明），`configs/default.yaml` 静态字段填齐
- [x] **1.2.2** `modules/dataset.py`：复用 `examples/ReactiveGWM/data/`，输出 `{video_latents [B,26,C,h,w], action [B,T_pix,K], prompt_str}`；CPU smoke 跑通（mini dataset）
- [x] **1.2.3** `diffsynth/pipelines/reactive_gwm_casual_forcing.py`：实现 `FlowMatchScheduler` wrapper + 1000-step 离散 + multi-step 采样 helper；verify `scheduler` 子命令过
- [x] **1.2.4** `diffsynth/models/reactive_gwm_casual_forcing_dit.py` — **TF dual-block 部分**：
  - 继承双向 `ReactiveGWMModel`
  - 移植 `_prepare_teacher_forcing_mask`，按 `(F_lat, frame_seqlen, num_frame_per_block, device)` 四元组缓存
  - TF 路径 forward：patchify `clean_x` / `noisy_x` → cat → per-frame timestep `[zeros ‖ t]` → flex_attention TF mask → 只取 noisy 半 unpatchify
  - state_dict strict 加载 `ReactiveGWM_base.safetensors` 0 missing / 0 unexpected
  - verify `state_dict` + `tf_mask` 子命令过
- [x] **1.2.5** `diffsynth/models/reactive_gwm_casual_forcing_dit.py` — **KV-cache 部分**：
  - 移植 CF 上游 `wan/modules/causal_model.py` KV-cache 数据结构（per-block + 每层 self-attn 一份 K/V buffer）
  - `init_kv_cache(batch_size, device, dtype, kv_window_size, sink_size, max_blocks)` → 返回 cache dict
  - `update_kv_cache(cache, layer_idx, new_K, new_V, block_idx)`：append 新 block + evict 超出 window 的旧 block（保留 sink + 最近 kv_window_size 个）
  - `refill_kv_cache(model, x_block, timestep, action, kv_cache)`：单 block forward 只写 K/V 不出 output（用于首帧 anchor + 已生成帧写入历史）
  - `forward(noisy_x_block, timestep, action, kv_cache=...)`：单 block forward 用 cache 算 attention 出 output（K/V 同步更新写回 cache）
  - cross-attn cache：prompt encoder 一次性算完写入 cache，所有帧 forward 直接读
  - verify `kv_cache_init` / `kv_cache_refill` 子命令过
- [x] **1.2.6** `model_fn_causal_forcing(tf_mode, ...)`：
  - `tf_mode=True`：编排 TF dual-block forward + 返回 noisy 半 flow 预测
  - `tf_mode=False, kv_cache=...`：编排 KV-cache 单 block forward + 返回 single-block flow 预测
  - verify `ar_tf` + `kv_cache_rollout`（用 KV-cache 跑 3-5 个 block）smoke 子命令过
- [x] **1.2.7** `modules/ar_tf.py`：`CFARTrainingModule.forward(data)`：
  - 取 26 帧 latent；
  - frame 0 anchor（I2V）：`noisy_x[:, 0] = clean_x[:, 0]`，timestep frame 0 = 0；
  - **逐帧独立 timestep**：每帧从全 `[0, 1000)` 范围独立采 index（对齐 CF 上游 `uniform_timestep=False`），同 index gather timestep / sigma / `training_weight`；
  - 调 `model_fn_causal_forcing(tf_mode=True, clean_x=x0, noisy_x=x0 + σ_f·ε, timestep=t_per_frame)`；
  - `loss = mean( MSE_per_frame · training_weight(t_f) )[:, 1:]`（per-frame BSMNTW 加权；frame 0 loss mask=0）
- [x] **1.2.8** `train.py`：
  - argparse（`--stage` / `--config` / 环境变量 `BASE_CKPT/DATA_ROOT/OUT/STUDENT_INIT/RESUME_FROM`）→ yaml deep-merge
  - accelerate `Accelerator` → 实例化 `CFARTrainingModule` → 单 AdamW（`lr=2e-6`, `betas=(0.0,0.999)`, `wd=0.01`；对齐 CF 上游） → gradient checkpointing on、bf16 mixed precision、`max_grad_norm=10`
  - **完整 checkpoint dump**（每 `save_steps`）：
    - bare model state（`stage1_ar.safetensors`）
    - optimizer state（`optimizer.bin`）
    - LR scheduler state（`scheduler.bin`）
    - Python / numpy / `torch` / `torch.cuda` RNG state（`rng.pt`）
    - step、epoch、wall-clock、global stats（`meta.json`）
    - accelerate `save_state(<dir>)`（DDP / FSDP 兼容）
    - 落盘到 `${OUT}/stage1_ar/step_<n>/`
  - **resume**：`--resume_from <step_dir>` 或 `RESUME_FROM=` env → `accelerate.load_state(<dir>)` + 恢复 step / RNG / scheduler；resume 后训练损失曲线与原 run 无可见 jump
- [x] **1.2.9** `launch/stage1_ar.sh`：8×H200 `accelerate launch` 命令；DDP 默认；env 覆盖路径
- [x] **1.2.10** **Sanity 推理（KV-cache 自回归）+ 后台 spawn 调度**：
  - `inference/fixed_sanity.py`：
    - 从 dataset clip 0 取 `prompt + action_序列`；
    - 若 action 序列像素帧数 < 1500，**用 idle action（10 维全零按键）填充到 1500**；
    - 若 > 1500 则截断到 1500；
    - 返回 `{prompt_str, action [1500, 10], first_frame_latent [1, C, h, w]}`
  - `inference/sanity_sample.py`：**KV-cache 自回归算法**（对齐 CF 上游 `pipeline/causal_diffusion_inference.py`）：
    1. 加载 ckpt 到 `CausalForcingReactiveGWMModel`；加载 VAE / T5 encoder；
    2. T5 编码 prompt → cross-attn cache 一次性写入；
    3. 初始化 KV-cache：`kv_window_size=16`，`sink_size=2`；
    4. `refill_kv_cache(first_frame_latent, t=0, action=action[0:K])` → 写入第 0 帧 K/V（K = 每 latent 帧对应的像素帧数）；
    5. 逐 latent 帧 rollout `i = 1, 2, ..., 300`：
       - 取本帧 action 切片 `action[i·K : (i+1)·K]`；
       - 从纯噪声 `noisy_x = randn(1, 1, C, h, w)` 起；
       - **multi-step flow-match**（默认 `diffusion_steps=50`）：每步 `model_fn_causal_forcing(tf_mode=False, noisy_x, timestep=t_i, action=...,  kv_cache=...)` → 按 FlowMatch 公式更新 `noisy_x`；
       - 去噪完成后得 `x0_i`；
       - `refill_kv_cache(x0_i, t=0, action=...)` → 写入 K/V（自动驱逐：保留 sink_size=2 + 最近 kv_window_size=16）；
       - append `x0_i` 到 output；
    6. VAE 解码 300 latent 帧 → ≈1500 像素帧 → ffmpeg 写 mp4（25 FPS，≈60 秒）
  - `inference/sanity_sample.py` 参数化：`--ckpt` / `--config` / `--out` / `--device {gpu_shared, cpu}` / `--diffusion_steps`（Stage 1/2 用 50，Stage 3 yaml 改 4）
  - `--device gpu_shared`：`os.environ["CUDA_VISIBLE_DEVICES"]="0"` + `torch.cuda.set_per_process_memory_fraction(0.2)`；`--device cpu`：纯 CPU bf16
  - `inference/spawn_sample.py`：
    - `spawn_sanity_sample(ckpt_path, step, out_dir, cfg) -> subprocess.Popen` → `Popen([sys.executable, "-m", "examples.ReactiveGWM_casual_forcing.inference.sanity_sample", ...])`，立即返回
    - main loop 维护 `last_proc`；若 `last_proc.poll() is None`（仍 alive） → log warning + skip 本次
    - subprocess stdout/stderr 重定向到 `${OUT}/stage1_ar/samples/step_<n>.log`
  - `train.py` 集成：每个保存点 `save_state()` 完成后 → `spawn_sanity_sample(ckpt, step, ...)` → 训练立即继续
  - verify `sanity_kv` 子命令：CPU 极小尺寸（5B 模型用 stub / 1 层 mini-DiT）端到端跑通 KV-cache rollout 5 帧 + mp4 输出
- [x] **1.2.11** `scripts/verify.py` 全套 CPU 子命令实现并通过
- [x] **1.2.12** **GPU dry-run**：8 卡 50–100 step，确认显存（预计 ~70-80 GB/GPU）、bf16 前后向无 NaN、loss 数值有限、save→spawn sanity 全链路通；sanity subprocess 用 `memory_fraction=0.2` 在 GPU 0 跑通；若 OOM 切 FSDP（决策 9，`auto_wrap_policy=None` 单 unit）
- [x] **1.2.13** **full run（✅ 完成；Updated 2026-06-06）**：`max_steps=50000`，`save_steps=2500`，sanity 每 save。最终 canonical = CF++ aligned 重训（`configs/stage1_ar.yaml` → `sf3_casual_forcing_2/stage1_ar`，最终 **step-24000**）。
  - **历史（旧 `sf3_casual_forcing/` 谱系，已被 aligned 重训取代）**：先 8 卡跑到 step-12500，2026-05-21 05:15 切到 4 卡（GPU 4-7）+ 旧 4card 配置（`grad_accum=2`，全局 batch 8）从 `state-12500` 无损 resume，跑到 ~step 14060；2026-05-21 SIGTERM 暂停于 state-17500（让出 GPU 给 Stage 2）。8→4 卡切换即一次实战 resume drill（DDP 全副本，无 jump）。后续用对齐修正版从 base 重训得到上面的 canonical step-24000。

### 1.3 测试与验证

`scripts/verify.py` 全套子命令全过（CPU：`state_dict` / `scheduler` / `tf_mask` / `ar_tf` / `kv_cache_init` / `kv_cache_refill` / `kv_cache_rollout` / `sanity_kv` / `smoke` + GPU：`bf16_two_paths`），覆盖：base ckpt strict 加载（0 missing/unexpected）、FlowMatch 逐点一致、TF mask 可见性（clean 半 causal、noisy↔prev-clean∪same-noisy）、逐帧独立 t + per-frame weight + frame0 anchor 排除、KV-cache sink+recent 驱逐、bf16 两路径数值有限+梯度通。

### 1.4 当前状态（✅ 完成；Updated 2026-06-06）

- 代码 + 全套 verify（1.2.1–1.2.12）全过；8×H200 dry-run 全链路 PASS（训练 + `save_state` + sanity subprocess + resume drill）。
- **canonical full run**：CF++ aligned 修正版从 base 重训 → `sf3_casual_forcing_2/stage1_ar`，最终 **step-24000**（`configs/stage1_ar.yaml`）。作为 Stage 2 的 student_init + teacher。
- **历史（旧谱系）**：旧 `sf3_casual_forcing/stage1_ar` 先 8 卡到 step-12500 → 2026-05-21 切 4 卡从 state-12500 无损 resume（实战 8→4 resume drill，DDP 全副本无 jump）；2026-05-21 SIGTERM 暂停于 state-17500。该谱系是旧 Stage 1 语义（commit `3546115`，uniform-across-frames t / [20,980] 截断 / 无 training_weight / beta1=0.9），已被上面的 aligned 重训取代。
- 启动见 `launch/stage1_ar.sh`（`NUM_PROCESSES` / `CUDA_VISIBLE_DEVICES` / `CFG` / `RESUME_STATE` env 覆盖）。

---

## Stage 2 — Causal Consistency Distillation（✅ 完成；Updated 2026-06-06）

> **状态（✅ 完成，Updated 2026-06-06）**：canonical run = `sf3_casual_forcing_2/stage2_cd`，student_init/teacher = Stage 1 step-24000，grad_accum=1，`cd_ema_shadow_device=cuda`，最终 step-20000；verify 11/11（含 `cd`，Stage 1 零回归）。resume 加 `RESUME_STATE=.../stage2_cd/state-<N>`。
>
> 下方为 Stage 2 的算法范围与设计约束（durable，复查 CD 对齐 / 重训时用）：
>
> **新增**：
> - `examples/.../modules/ema.py`（轻量 EMA，`decay=0.99` **从 step 0 更新**，Stage 2 / 3 共用）
> - `examples/.../modules/cd.py`：`CFCDTrainingModule`（generator + EMA-student 目标网络 + 冻结 Stage 1 teacher + 冻结 encoders + **negative-prompt uncond 编码**；`forward(data)` 返回 CD MSE：**uniform t、x0 一致性、teacher CFG 单步 ODE、frame-0 anchor 排除**）
> - `examples/.../configs/stage2_cd.yaml`（`discrete_cd_N: 48`、`cd_guidance_scale: 3.0`、`student_init: <stage1 产物>`、`negative_prompt`）
> - `examples/.../launch/stage2_cd.sh`
>
> **追加到 `diffsynth/pipelines/reactive_gwm_casual_forcing.py`**：
> - CD step helper（teacher 单步 ODE + **negative-prompt CFG** `v_uncond + 3·(v_cond − v_uncond)`）
> - **flow→x0 转换**（student/EMA 的 `cm_pred = latent − σ·flow`；CD 比的是 x0 预测，不是 flow。上游 `WanDiffusionWrapper.forward()` 会返回 `(flow, x0)`，但本项目 **`model_fn_causal_forcing()` 默认只返回 flow**，Stage 2 必须在 CD helper 内部做转换，不改默认返回类型）
> - `cd_discrete_N=48` 离散调度路径（`set_timesteps(48)`，extra_one_step 丢 σ=0 → **48 节点，非 49**）
>
> **追加到 `train.py`**：
> - `--stage cd` 派发分支（仍单优化器、单 backward）
> - EMA dump/load 进 checkpoint
>
> **追加到 `scripts/verify.py`**：
> - `cd` 子命令：验证 **上游语义等价** 的 `cm_pred`（上游 forward 第 2 返回值是 x0；本项目保持 forward/model_fn 默认返回 flow，并在 CD helper 中用 `x0 = latent − σ·flow` 得到 cm_pred）；student/EMA-student 同初始化时 `cm_pred_t ≈ cm_pred_t_next` → loss → 0；teacher 单步 ODE = `latent_t − dt·v_cfg`，`dt=(t−t_next)/1000`（手算等价）；negative-prompt CFG 路径正确；uniform t（全帧同 t）+ frame-0 anchor 排除出 loss
>
> **不再追加**（Stage 1 已实现）：
> - `CausalForcingReactiveGWMModel` 的 TF dual-block 与 KV-cache 路径（Stage 2 训练直接用 TF；sanity 推理直接用 KV-cache）
> - `sanity_sample.py` 算法（沿用 Stage 1 实现，diffusion_steps 默认 50）
>
> **关键算法点**（对齐上游 `naive_consistency.py` + I2V，详见 PLAN §2 Stage 2）：
> - **uniform t**（单 t broadcast 全帧；**≠** Stage 1 逐帧独立）；CD 调度 **48 节点**（非 49）
> - `cm_pred` = **x0 预测**（上游取 forward 第 2 返回值；本项目不改默认返回，统一在 CD helper 内用 `latent − σ·flow` 转换）；CD loss = `MSE(cm_t, cm_t_next)` 纯 MSE，**无 training_weight**
> - teacher（冻结 Stage 1）单步 ODE + **negative-prompt CFG（`guidance_scale=3.0`，必须）** 生成 `latent_t_next`
> - EMA-student（目标网络，**从 step 0** 更新，`decay=0.99`）算 `cm_pred_t_next`；`ema_start_step=200` 仅决定存档格式
> - **I2V**：frame 0 anchor（`latent_t` / `latent_t_next` 首帧 = clean、t=0），CD loss **排除 frame 0**
> - 3×5B 同驻（teacher / EMA 冻结、无优化器）；DDP 优先，OOM 切 FSDP
>
> **训练规格**（同 [通用训练规格](#通用训练规格stage--1--2--3-共用可在各-stage-yaml-覆盖)）：
> - 数据：`metadata.csv` 全量
> - 训练窗口：26 latent 帧
> - 步数：**50000**
> - save：每 2500 step（包含 EMA state；最终**导出优先 EMA 权重**）
> - sanity 推理：每 2500 step **后台 spawn 60 秒视频**（沿用 Stage 1 的 KV-cache 自回归路径，**用 EMA 权重**做推理）；不阻塞训练
> - resume 完整
>
> **产物**：`stage2_cd.safetensors`（**EMA 权重优先导出**，bare `CausalForcingReactiveGWMModel`）+ 20 个中间 ckpt + 对应 sanity mp4

#### ⚠️ Stage 2 实施前置约束（保护规则——不破坏正在跑的 Stage 1）

> 1. **不改 `model_fn_causal_forcing()` 默认返回类型**：Stage 1 假设它返回单个 flow tensor（`modules/ar_tf.py`）。要 x0 就在 CD helper 里算 `x0 = latent − σ·flow`，或加 `return_x0=False` 可选参数，默认保持 flow。
> 2. **不改 `CausalForcingReactiveGWMModel.forward()` 默认返回类型**：保持返回单 tensor，**不要**改成 `(flow, x0)`；x0 转换放 CD helper / model_fn optional 分支。
> 3. **不动 Stage 1 的 AR timestep / training_weight / loss 逻辑**：`modules/ar_tf.py` 已是逐帧独立 t + per-frame weight + frame0 anchor；Stage 2 是另一条 `cd.py` 路径（uniform t，不复用）。
> 4. **`train.py` 只把 `--stage cd` 的 `raise NotImplementedError`（`train.py:78-79`）换成 `_run_cd(args, cfg)` + 新增该函数**；不动 `_run_ar_tf()` 主循环。
> 5. **Stage 2 改完后必须重跑全套 Stage 1 verify**：`state_dict scheduler tf_mask ar_tf kv_cache_init kv_cache_refill kv_cache_rollout sanity_kv smoke bf16_two_paths`，确认 Stage 1 路径零回归。
>
> **flow→x0 转换（确定用此式）**：`x0 = latent_t − σ·flow_pred`，σ 用与加噪一致的 **per-frame sigma**（frame0=0，使 `x0[:,0]=latent[:,0]=clean`）；σ 来自 CD 调度 `sigmas[idx]`（teacher 推进后那次用 `sigmas[idx+1]`）。
>
> **frame-0 anchor 完整机制（I2V）**：要 anchor **两个** latent——`latent_t[:,0]=x0[:,0]`+`t[:,0]=0` 与 `latent_t_next[:,0]=x0[:,0]`+`t_next[:,0]=0`（teacher 单步 ODE 后首帧仍 = 给定 GT）；再 `loss=MSE(cm_t[:,1:], cm_t_next[:,1:])` 排除 frame0。只排 loss、不 anchor `latent_t_next` 首帧 = 与 I2V 语义不符。
>
> **项目级适配（非 CD 算法差异，但全程在场）**：① per-block **action 条件**（上游 Wan 无）——CD 一致性是"给定同一 prompt+action"下的 x0 一致性；② **26 帧 / 48 通道**（Wan2.2 VAE，vs 上游 21 帧 / 16 通道）。

---

## Stage 3 — Asymmetric DMD self-forcing（✅ 完成；Updated 2026-06-06）

> **状态（✅ 完成，Updated 2026-06-06）**：26-window canonical run → `sf3_casual_forcing_2/stage3_dmd` step-11400；长程 rollout 延伸 run `stage3_dmd_long_from11200` 进行中。verify 13/13 + 训练核心 / save / sanity / resume drill 全打通（显存与 FSDP 方案见 §3.2.7）。
> **范围概览已按 [stage3_cf_alignment_review.md](./stage3_cf_alignment_review.md) 订正**（上游真实文件见报告 §1；sink / 26 帧 / same_step / EMA 等决议见 §0.1）。
> **关键自决项（durable，复查 DMD 对齐时用）**：
> 1. **frame0 在 DMD score 的加噪（相对上游的必要 I2V 适配）**：上游 teacher/critic 是 T2V、对全 21 帧加同一 uniform 噪声；本项目 teacher/critic 是**双向 I2V base**，走 `model_fn_wan_video`（`fuse_vae_embedding_in_latents=True` 会**强制 frame0 timestep=0**）。故 DMD score / critic 加噪只作用 **frame 1..25**、frame0 保持 clean anchor；flow→x0 用 per-frame σ（frame0=0，且 Wan flow-match 下 **σ=t/1000** 精确成立，直接算不用 argmin）。frame0 在 DMD loss / critic loss / normalizer 贡献恒为 0，自洽。
> 2. **backward 显存策略**：先完全按上游"**整窗 26 帧一次 backward**"写，不写 per_frame_backward/max_dmd_frames 脚手架；dry-run 真 OOM 再加（届时标注项目适配）。靠 gradient checkpointing + 只有 exit-step forward 进图控显存。
> 3. **训练 rollout = DMD2 backward-simulation（一致性采样器）**：每步去噪到 x0 后用**全新随机噪声**重加噪到下一 timestep（`add_noise`），**≠** sanity 推理的确定性 ODE step（`x + flow·Δσ`）。需在 pipeline 单独实现，不复用 `sample_multistep_kv_cache`。
> 4. **EMA 仅导出**：step-200 懒创建、只在 generator step（每 5 全局步）后更新、fp32 shadow（复用 `ema.py`），导出 cast bf16；**不进任何 loss**（与 Stage 2 的"in-loss 目标网络、step 0 起"不同，二者独立）。
> 5. **teacher/critic = 双向 `ReactiveGWMModel`**（父类，非 CF 子类），都从 `base_ckpt` 起步；teacher 冻结、critic 可训练副本；用 base 原生 `model_fn_wan_video` 打分（返回 flow→转 x0）。
>
> 范围概览（详见 PLAN.md §2 Stage 3 + §3）：
>
> **追加到 `diffsynth/pipelines/reactive_gwm_casual_forcing.py`**：
> - holistic DMD helpers（**x0 空间**：`x0=latent−σ·flow`；teacher CFG score `real_guidance_scale=3.0` / critic x0→flow / normalizer `mean(abs(generated_x0−real_x0))` 无 clamp+`nan_to_num` / `loss_gen = 0.5·MSE(generated_x0, (generated_x0−grad).detach())`；上游 `model/dmd.py`）
>
> **新增**：
> - `examples/.../modules/dmd.py`：`CFDMDTrainingModule`（generator + EMA（仅导出）+ **双向** teacher 冻结 + **双向** critic 可训练 + 冻结 encoders；4-step self-rollout、**26 帧定长窗口**（frame0 GT anchor 不算 loss + 25 生成）、训练 rollout **全量 buffer 不驱逐**、`same_step_across_blocks=true`、`stochastic_exit_step=true`（每次 rollout 随机 exit；因 `same_step_across_blocks=true`，所有 block 共享该 exit，不是每 block 独立采样）、`detach_history_kv=true`（中间 step+context 更新 no_grad）；**梯度 = shared exit-step + 窗口 mask，不是 `grad_horizon=8`**）（上游 `pipeline/self_forcing_training.py` + `model/base.py`）
> - `examples/.../modules/runner_dmd.py`：双优化器交替循环（上游 `trainer/distillation.py`：`dfake_gen_update_ratio=5`=generator 每 5 步 / critic 每步、`lr=2e-6` generator(`beta1=0.0/beta2=0.999`) / `lr_critic=4e-7` critic(`beta1_critic=0.0/beta2_critic=0.999`)、grad clip 10、EMA `decay=0.99` **仅导出**+懒创建 `ema_start_step=200`+只在 generator step 更新、DDP / FSDP 切换）。⚠️ `critic_update_mode=per_frame_backward` 是**项目显存适配候选**（上游整窗一次 backward，review §3.6），待显存实测后定
> - `examples/.../configs/stage3_dmd.yaml`（`sanity_sample_diffusion_steps: 4`；`dmd_score_frames: 26`、`same_step_across_blocks: true`、`ts_schedule: false`、`denoising_loss_type: flow`、`beta1_critic/beta2_critic`、DMD timestep clamp `[20,980]`；详见 PLAN §7）
> - `examples/.../launch/stage3_dmd.sh`（含 `--use_fsdp` 兜底）
>
> **追加到 `train.py`**：
> - `--stage dmd` 派发到 `runner_dmd.py` 独立训练循环
> - 三模型（generator / teacher 双向 frozen / critic 双向 trainable）+ EMA 都要 checkpoint dump/load
>
> **追加到 `scripts/verify.py`**：
> - `rollout` 子命令：4 步逐帧 self-rollout；shared random exit step（`same_step_across_blocks=true` 时所有 block 共享同一个随机 exit，只有 exit-step forward 进计算图）；中间 step + context-cache 更新 `no_grad`（`detach_history_kv`）；帧维只在 26 帧窗口（除 frame0）保留梯度（**非** `grad_horizon=8`）；训练 rollout 全量 buffer 不驱逐
> - `dmd` 子命令：**DMD grad 在 x0 空间**（`x0=latent−σ·flow`，`grad=fake_x0−real_x0`，`normalizer=mean(abs(generated_x0−real_x0))`）；timestep uniform+shift+clamp `[20,980]`；`same_step_across_blocks=true`；teacher CFG / critic flow score 形状对齐；`loss_gen` 反传只到 generator；**critic flow loss 与 DMD x0 loss 不同空间**、critic loss 反传只到 critic；frame0 anchor mask；EMA 仅导出、只覆盖 `requires_grad=True`；DDP-safe phase 切换
>
> **不再追加**（Stage 1 已实现）：
> - `CausalForcingReactiveGWMModel` 的 KV-cache 路径（Stage 3 self-rollout 直接复用 Stage 1 已实现的 KV-cache forward）
> - `model_fn_causal_forcing(tf_mode=False)` 分支（Stage 1 已实现）
> - `sanity_sample.py` 算法（沿用 Stage 1 实现，只通过 yaml 把 `sanity_sample_diffusion_steps` 从 50 改成 4）
>
> **训练规格**（同 [通用训练规格](#通用训练规格stage--1--2--3-共用可在各-stage-yaml-覆盖)）：
> - 数据：`metadata.csv` 全量（Stage 3 实际只需 prompt + action + 首帧 latent）
> - rollout 窗口：**26 帧定长**（`dmd_score_frames=26`，frame0 GT anchor 不算 loss + 25 生成；上游 `num_training_frames` 是变长 rollout 上限，本项目定长，review §3.5）
> - 步数：**50000 上限**（CF++ 论文 Stage 3 = 1K 步 @ batch 64；本项目 batch 4-8 小 8-16×，按样本曝光 ≈ 论文 8-16K 步；**保留 50K 上限但明确靠早停**：DMD self-rollout 易崩，每 2500 step sanity 盯崩溃 / plateau → 早停，50K 是 cap 不是必跑完）
> - save：每 2500 step（含 generator / critic / EMA 全部 state）
> - sanity 推理：每 2500 step 后台 spawn **60 秒 4-step KV-cache 视频**；不阻塞训练
> - resume 完整
>
> **产物**：`stage3_dmd_generator.safetensors`、`stage3_dmd_ema.safetensors`（**最终导出**）、`stage3_dmd_critic.safetensors` + 20 个中间 ckpt + 对应 sanity mp4

> ⚠️ **FSDP 存点 / Resume 拓扑语义（2026-05-25 补；实战于 8→4 reshard）**
>
> Stage 3 实际后端 = **FSDP `FULL_SHARD` + `use_orig_params=true` + `SHARDED_STATE_DICT`**（`launch/stage3_dmd.sh`，`USE_FSDP=1`）。存点分两类，**reshard 能力不同**：
>
> - **模型 + 优化器**（`accelerator.save_state` → DCP 分布式 checkpoint）：**可跨 world_size reshard**。8 卡存的 `state-N/` 能在 4 卡 resume —— 这正是 8→4 reshard 成功的原因（`SHARDED_STATE_DICT` 走 `torch.distributed.checkpoint`，与卡数解耦）。
> - **EMA fp32 shadow**（`runner_dmd.py::_save_ema_state` → 每 rank 一个 `ema-rank{NNNNN}.safetensors`，**按 rank 分片、非 DCP**）：**不能 reshard**。改卡数（或 DDP↔FSDP 切换）时，旧 world_size 的 N 个 rank-shard 无法映射到新 world_size → `_load_ema_state` 找不到匹配文件 → **EMA 从当前 generator 权重重新累积**（decay 0.99，约几百个 generator-step 重收敛）。
>   - ⚠️ 只丢"平均历史"，**不丢已导出的 EMA 快照**：`step-N.safetensors` / `stage3_dmd_ema.safetensors` 是导出时刻的 EMA 权重快照，推理产物不受影响。
>   - 实战：2026-05-24 的 8→4 reshard 即触发 EMA 重置（且 `state-4800` 是旧 runner 产物、本就无 `ema-rank*` 文件，无论如何都会重置）。
>
> **操作建议**：想保住 EMA 平均历史 → **resume 时保持 world_size 与存点一致**（同卡数、同 FSDP 模式）。必须改卡数时，接受 EMA 重置（不影响最终产物质量，只是 EMA 平滑窗口从 resume 点重新起算）。
> **若将来要让 EMA 也能 reshard**：把 shadow 纳入 DCP（随 `save_state` 一起 `torch.distributed.checkpoint`），或存时 gather 成 full-state、load 时按新 world_size 重分发。
>
> 关联：grad-clip / idle-group 的 FSDP 修复见文末「变更日志」2026-05-25 `S3-fsdp-fix` 条 + `scripts/fsdp_idle_grad_probe.py`。

### 3.0 目标

训练出 **`stage3_dmd_ema.safetensors`**：从 Stage 2 产物（CD EMA 权重）出发，用 **asymmetric DMD self-forcing**（4-step 因果 KV-cache rollout + 双向 teacher/critic 分布匹配）把 generator 蒸馏成 **4 步逐帧自回归**生成器。产物质量目标 = 4 步因果生成与双向 base 对齐（动作响应 + 长程一致性）。

> ⚠️ **DMD self-rollout 易崩**：50K 是 cap 不是必跑完，明确靠"每 2500 step sanity 盯崩溃 / plateau → 早停"兜底（PLAN §2 Stage 3 论文对照）。

### 3.1 范围（本 Stage 新增/修改的工件）

**`diffsynth/pipelines/reactive_gwm_casual_forcing.py` 追加**（自包含；不改 Stage 1/2 任何函数 / 不改 `model_fn` / `forward` 默认返回 flow）：

- [x] `warp_denoising_steps(sched1000, denoising_step_list, device)`：上游 warp（`timesteps[1000 − step_list]`）→ 返回 warped timesteps + sigmas（4 步）。
- [x] `dmd_self_rollout(...)`：**DMD2 backward-simulation** KV-cache self-rollout。26 帧定长（frame0=GT anchor 先 refill）；逐帧 4-step：shared random `exit_idx`（`same_step_across_blocks=true`），非 exit 步 `no_grad` 去噪到 x0 + **全新噪声重加噪到下一步**（一致性采样器），exit 步 forward（`keep_grad=True` 时保留 grad）→ x0 → break；refill_kv_cache（`denoised.detach()`，no_grad）提交历史；全量 buffer 不驱逐（`sink_size=0`、window≥26）。**返回 `(video [B,26,C,H,W], exit_idx)`**——frame0=anchor（无梯度），生成帧 1..25 带梯度；frame0 排除不靠 boolean mask，而在 loss helper 内用 `[:,1:]`（`first_frame_anchor`）。
- [x] `dmd_score_x0(score_model, noisy_bfchw, t_scalar, context, action, sigma_t)`：双向 teacher/critic 打分——`model_fn_wan_video`（permute [B,C,F,H,W]，`fuse_vae_embedding_in_latents=True`，frame0 自动 t=0）→ flow → `x0 = noisy − σ_pf·flow`（per-frame σ：frame0=0、其余=sigma_t）。core = `_dmd_score_flow`（critic 训练传 `use_gradient_checkpointing`）。
- [x] `dmd_distribution_matching_loss(...)`：x0 空间——teacher CFG `real_x0 = cond + g·(cond−uncond)`（g=3）/ critic cond-only（`fake_guidance=0`）/ `grad = fake_x0 − real_x0` / `normalizer = mean(abs(pred_x0 − real_x0))[1,2,3,4] keepdim` 无 clamp（**frame0 排除**避免 0 稀释）/ `nan_to_num` / `loss = 0.5·MSE(pred_x0[:,1:].double(), (pred_x0 − grad)[:,1:].detach().double())`。score 全程 `no_grad`，只有 `pred_x0`（rollout 产物）带 grad。
- [x] `dmd_critic_flow_loss(...)`：critic 在 generated（detach）上加 uniform 噪声（frame 1..25）→ critic flow 预测 → flow-matching MSE `mean((critic_flow[:,1:] − (noise − generated)[:,1:])²)`（= 上游 x0→flow→FlowPredLoss 在 flow-native 模型下的等价式）。
- [x] `sample_dmd_timestep(...)`：uniform 采样 `[0,1000)` → warp shift=5.0 → clamp `[20,980]`；σ=t/1000。+ `_sample_shared_exit_index`（DDP `dist.broadcast` 同步）。

**`examples/ReactiveGWM_casual_forcing/` 新增**：

- [x] `modules/dmd.py`：`CFDMDTrainingModule`（generator=`pipe.dit` 因果可训 + 双向 critic **子模块**可训 + 双向 teacher `_aux` 冻结 + EMA shadow 仅导出 + 冻结 encoders + neg context）。`forward(data, phase)` 按 `phase in {generator, critic}` 返回对应 loss；`generator_params()`/`critic_params()` 给 runner 建双优化器；`maybe_update_ema(step)`/`dmd_export_generator(step, prefer_ema)`/`dmd_export_critic()`/`ema_full_state_dict`/`load_ema_full_state`。
- [x] `modules/runner_dmd.py`：双优化器交替（generator 每 `dfake_gen_update_ratio=5` 步 / critic 每步）。**架构决定：整个 module 仅 DDP 包一次 + `find_unused_parameters=True`**（每 phase 只用 gen 或 critic 之一；module 每 phase 调一次 → generator 内部 25 次 sub-forward 不触发多次 DDP-forward 的 reducer 问题）；teacher 冻结非子模块不进 DDP；EMA step-200 懒创建+只在 generator step 更新；checkpoint = accelerate state（gen+critic+双优化器）+ `state-N/ema.safetensors`（EMA fp32 shadow，非 param 单独存）+ `step-N.safetensors`（generator EMA 导出，sanity/下阶段）；`USE_FSDP=1` 切换。
- [x] `configs/stage3_dmd.yaml`（+ `_dryrun.yaml` / `_drill.yaml`；**无 4card yaml**——Stage 3 runner 固定 grad_accum=1，卡数由 launch `NUM_PROCESSES` 控制，全局 batch=卡数）：`stage: dmd`、`student_init=<stage2 EMA 产物，占位 step-50000，STUDENT_INIT env 覆盖>`、`max_steps=50000`、`save_steps=2500`、`sanity_sample_diffusion_steps: 4`、`dmd_score_frames: 26`、`same_step_across_blocks: true`、`ts_schedule: false`、`denoising_loss_type: flow`、`real_guidance_scale: 3.0`、`fake_guidance_scale: 0.0`、`dmd_timestep_clamp: [20,980]`、`dfake_gen_update_ratio: 5`、`lr_critic: 4e-7`、`beta1_critic/beta2_critic`、`dmd_ema_start_step: 200`、`dmd_ema_shadow_device: cpu`（3×5B 显存紧；EMA 每 5 步才更新摊薄 CPU 传输）、训练 rollout buffer `dmd_train_kv_window: 26` / `dmd_train_sink: 0`。
- [x] `launch/stage3_dmd.sh`（`accelerate launch train.py --stage dmd`；env `STUDENT_INIT/BASE_CKPT/OUT/RESUME_STATE/NUM_PROCESSES/MAIN_PORT(29522)`；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments`；`USE_FSDP=1` 切 FSDP 单 unit 兜底）。

**`train.py` 追加**：[x] `--stage dmd` 派发到 `_run_dmd`（调 `runner_dmd.run_dmd`，替换 `NotImplementedError`）。

**`scripts/verify.py` 追加**：[x] `rollout` / `dmd` 子命令（见 3.3）+ `_build_tiny_bidir`（双向 teacher/critic tiny stub）。

> 不再追加（Stage 1/2 已实现）：`CausalForcingReactiveGWMModel` KV-cache forward（rollout 直接复用 `_forward_kv_cache`/`refill_kv_cache`）、`sanity_sample.py` 算法（沿用，yaml 把 `sanity_sample_diffusion_steps` 设 4）、`ema.py`（复用 fp32 shadow）。

### 3.2 实现子步骤

- [x] **3.2.1** pipeline 追加 DMD helpers（warp / timestep / rollout / score_x0 / dmd_loss / critic_flow_loss）+ `_sample_shared_exit_index`；不动 Stage 1/2 函数。`py_compile` ✓
- [x] **3.2.2** `modules/dmd.py`：`CFDMDTrainingModule` —— 加载 generator（CF 子类，Stage 2 起步）+ 双向 teacher/critic（父类，base_ckpt）+ EMA shadow + neg context；`forward(data, "generator")` = rollout + DMD loss；`forward(data, "critic")` = no_grad rollout + critic flow loss；frame0 anchor 全程排除。`py_compile` ✓
- [x] **3.2.3** `modules/runner_dmd.py`：双优化器交替 + EMA（step-200 懒创建、generator step 后更新）+ save/resume（generator+critic+EMA+accelerate state）+ sanity spawn（4-step）。`py_compile` ✓
- [x] **3.2.4** `configs/stage3_dmd*.yaml`（yaml+merge 校验过）+ `launch/stage3_dmd.sh`（`bash -n` 过）+ `train.py::_run_dmd` 派发。
- [x] **3.2.5** `scripts/verify.py` 的 `rollout` / `dmd` 子命令 + `_build_tiny_bidir` 实现。`py_compile` ✓ —— **运行验证已过，见 3.2.6**。
- [x] **3.2.6 ✅（2026-05-22, GPU 0, `CUDA_VISIBLE_DEVICES=0`）** 运行全套 verify（CPU：`state_dict scheduler tf_mask ar_tf kv_cache_init kv_cache_refill kv_cache_rollout sanity_kv smoke cd` + **新 `rollout` `dmd`** + GPU：`bf16_two_paths`）—— **13/13 全过**：`rollout`(warp σ=t/1000✓ / exit_idx 共享✓ / frame0 anchor✓ / keep_grad→gen✓) + `dmd`(timestep clamp[20,980]✓ / x0-score frame0 anchor✓ / gen-loss(x0)→generator only✓ / critic-loss(flow)→critic only✓) 通过，Stage 1/2 **零回归**。`diffsynth_sf` CUDA 环境。
- [x] **3.2.7 ✅ 训练 / save / sanity / resume drill 打通（2026-05-23）**：NO_WRAP/FULL_SHARD + 首帧 VAE 修复 + teacher CPU offload + exit-step checkpoint（KV-cache read-state snapshot）后，**26 帧完整窗口**在 GPU0-3（有邻居进程占 ~21GB/卡）完成 2-step gen+critic 训练，loss finite、无 NaN/OOM；8 帧 memdiag 完成 4-step；targeted save→spawn sanity→resume drill 全链路通过。
- [x] **3.2.8 ✅ full run（Updated 2026-06-06）**：`max_steps=50000` 上限 + 每 save_steps sanity 盯崩溃 → 早停兜底。canonical 26-window run → `sf3_casual_forcing_2/stage3_dmd`，跑到 **step-11400**（`configs/stage3_dmd.yaml`，student_init=stage2 step-20000）。长程 rollout 延伸 run（`configs/stage3_dmd_long_from11200.yaml`，generator 从 step-11200 起，rollout 到 101 帧只 score 末尾 26 帧）→ `sf3_casual_forcing_2/stage3_dmd_long_from11200`，**进行中**（截至 2026-06-06 step-4000）。

> **3.2.6 verify 13/13 全过（2026-05-22）；3.2.7 训练核心 + save/sanity/resume drill 打通（2026-05-23）；3.2.8 full run ✅ 完成（26-window step-11400 + 长程延伸 run 进行中，Updated 2026-06-06）。**

#### 3.2.7 显存 / FSDP 方案（2026-05-23 定稿，已落在工作树）

经多轮 dry-run 实测定下的显存方案（均不改 DMD 数学语义）：

- **FSDP NO_WRAP / FULL_SHARD 单 unit**：per-Model / per-block auto-wrap 会让 generator 的 KV-cache 方法（`refill_kv_cache` / `_patchify_and_geom` 等绕过 `__call__` 的入口）读到分片碎片（`block.modulation` shape err / conv1D）；`summon_full_params` 也 hold 不住 25 帧 rollout 的多入口循环 → **NO_WRAP 是唯一可行的 FSDP**（forward 时整模型一起 gather，参数完整）。
- **首帧-only VAE encode**（`_resolve_inputs` 只 encode `data["video"][:1]` + `no_grad`）：避免整段 101 帧 VAE encode 的 ~50GB window-independent 浪费。
- **teacher CPU offload**：冻结 teacher 常驻 CPU，仅 generator DMD score 时上卡、完后搬回。
- **exit-step activation checkpoint + KV-cache read-state snapshot**（共享 K/V buffer、只 clone end index）：压 26 帧整窗 backward 激活峰值，不改 rollout 数学。
- **单 AdamW 两 param group**（gen / critic 各自 lr/betas，每 phase 单独 backward+step）：等价上游双优化器交替，且 `accelerator.save_state` 能存 optimizer。
- **`FSDP_IGNORED_MODULES=pipe.text_encoder|pipe.vae`** 排除冻结 T5/VAE 出 root flat-param；**export 在 `summon_full_params(rank0_only, offload_to_cpu)` 内**（否则写出 shape `[0]` shard，sanity strict load 报错）；`mixed_precision=no`（省 fp32 master）。

**实测（均不杀已有进程）**：verify 13/13 PASS；8 帧 memdiag 4-step PASS、26 帧完整窗口 2-step PASS（loss finite、无 NaN/OOM）；save → sanity（`memory_fraction=0.45` 4-step CFG rollout）→ `state-1` resume → step2 全链路 PASS。FSDP 存点 reshard 语义见上方 caveat。

**当前结论：** Stage3 训练核心（26 帧 rollout + generator DMD x0 loss + critic flow loss + 上游语义 generator/critic 交替）与 save/sanity/resume drill 均打通，功能仍与上游 Causal-Forcing++ Stage3 对齐。

### 3.3 测试与验证

`scripts/verify.py` 新增 `rollout` / `dmd`（tiny stub）+ 复用 Stage 1/2 全套，**13/13 PASS**（GPU 0）：`rollout`（warp σ=t/1000、shared exit、只 exit-step 进图、frame0 anchor、全量 buffer 不驱逐）、`dmd`（x0 空间 grad `x0=noisy−(t/1000)·flow` / `grad=fake−real` / normalizer 无 clamp、timestep clamp `[20,980]`、teacher CFG g=3 / critic cond-only、gen-loss→generator only / critic-loss→critic only、frame0 mask）。真实 5B smoke（Stage2 step-12500 generator + base critic/teacher）在 8 帧 4-step / 26 帧 2-step memdiag 下 loss finite、梯度可反传、无 NaN。

### 3.4 验收（✅ 完成；Updated 2026-06-06）

canonical 26-window run 到 step-11400 + 长程 rollout 延伸 run（`stage3_dmd_long_from11200`，进行中），双优化器交替 / EMA 懒创建 / save+resume / 4-step sanity 全链路打通，verify 13/13 + Stage 1/2 零回归（见 §3.3）。产物 `step-N.safetensors`（generator EMA 权重）可被 `CausalForcingReactiveGWMModel` strict 加载；跑满 `max_steps` 收尾才额外写 `stage3_dmd_{generator,ema,critic}.safetensors` 三件套（`runner_dmd.py:438/446/454`）。

## Stage 4 — 推理与评测（🔒 Locked）

> 等 Stage 3 review 通过后才展开此节子任务。
> 当前仅提前预告 **范围概览**（详见 PLAN.md §8）：
>
> **新增**：
> - `examples/.../inference/causal_inference.py`：4 步 KV-cache 推理核心独立 CLI（与训练中 spawn 的 sanity 路径功能等价，但允许批量评测的更稳定接口；可承担多 prompt / 多 action 配置的并行推理）
> - `examples/.../inference/infer.py`：单 clip CLI（接受任意 prompt + action + 任意长度，输出 mp4）
> - `examples/.../inference/eval_action.py`：13 动作轴网格 + 区域 motion energy + MSE；评测 metric helpers importlib file-path 从 `examples/ReactiveGWM` 加载（避免 `inference` 包名冲突）
> - `examples/.../launch/infer.sh`
>
> **不再追加**（Stage 1 已实现）：
> - `sanity_sample.py` 已经是完整的 KV-cache 自回归推理，Stage 4 只是把它"产品化"成独立 CLI + 评测脚本
>
> **Acceptance**：
> - 单 clip 4-step 推理产出 60 秒视频，肉眼检查无明显伪影
> - 13 动作轴评测 vs 双向 base：每个动作轴产生对照视频 + 区域 motion energy + MSE 报告
> - 推理速度记录（FPS / 单帧延迟），v1 不作为硬指标

---

## 变更日志（每次完成子任务时追加一行）

格式：`YYYY-MM-DD  <stage>.<sub>  <slug>  ✅`

- 2026-05-20  Stage1  代码 + 10 verify 子命令全过；8×H200 dry-run 全链路 PASS（训练 + save_state + sanity subprocess + resume drill；含 RoPE / unpatchify / sanity-OOM / mp4-writer / resume-num_steps 等 fix）。✅
- 2026-05-21  Stage1  对齐 CF 上游 4 项（逐帧独立 t / per-frame weight / 全 [0,1000) / AdamW betas=(0,0.999)+wd=0.01）；full run 8 卡→4 卡（GPU4-7，grad_accum=2）从 state-12500 无损 resume；SIGTERM 暂停于 state-17500。⚠️ 已落盘 step-15000 等 ckpt 仍是旧 Stage1 语义（commit 3546115），严格对齐需重启重训。
- 2026-05-21  Stage2  全代码 + verify 11/11（Stage1 零回归）+ 4 卡 dry-run / resume drill；EMA 改 fp32 master shadow（对齐上游 EMA_FSDP，修 bf16 累加吃增量）；sanity KV-cache 补 CFG + 步数→4（=warped [1000,750,500,250]）；50K run GPU4-7 独占（cd_ema_shadow_device=cuda ~5.85s/step）。🟡 teacher=step-15000 未收敛，质量受限。
- 2026-05-22  Stage3-doc  按 stage3_cf_alignment_review.md + CF++ 论文订正设计（上游真实文件、sink、26 帧、same_step=true、DMD x0 空间 + normalizer、ts_schedule=false、EMA 仅导出懒创建、4-step warp 逐点等价、5B vs 论文 14B、50K cap+早停）。✅
- 2026-05-22  Stage3-code  全代码：pipeline DMD helpers + modules/dmd.py + runner_dmd.py + configs + launch + train.py::_run_dmd + verify rollout/dmd。✅
- 2026-05-22  Stage3  verify 13/13 PASS（新 rollout/dmd + Stage1/2 零回归）。✅
- 2026-05-23  Stage3  训练核心 + save/sanity/resume drill 打通（显存/FSDP 方案见 §3.2.7）：NO_WRAP/FULL_SHARD、首帧 VAE 修复、teacher CPU offload、exit-step checkpoint + KV snapshot、单 AdamW 两 group、summon export；8 帧 4-step / 26 帧 2-step PASS。✅
- 2026-05-25  S3-fsdp-fix  代码审查发现 Stage 3 runner 在 **FSDP** 下两个 silent 问题并修复（`runner_dmd.py`，仅下次启动生效）：①**grad-clip 传子集**（`generator_params()`/`critic_params()`）→ accelerate 回退普通 clip → 在分片梯度上只算**单-rank 局部范数**（阈值≈10·√world_size、各 rank 不一致）；改为两 phase 都传 `model.parameters()` 全集 → 走 FSDP 感知跨-rank 范数（Stage 1/2 本就这样传，无回归）。②**单-optimizer 两-group 依赖"idle 组 grad=None 不更新"**，该前提 DDP/单卡成立、但 FSDP 单 flat-param 下 idle 组可能拿到 grad=0 → 被 AdamW 每步 weight-decay + 二阶矩衰减（generator 有效 lr 漂移 ~2.4×）；改为每 phase 在 step 前 `_none_out_grads(idle 组)`，三种后端都恢复"只更新活跃组"。新增 `scripts/fsdp_idle_grad_probe.py`（玩具模型 1-/2-卡探针，判定本 FSDP 版本给 idle 组 None vs 0、并验证修复）。py_compile 过；②是否真在咬见下条探针实测。✅
- 2026-05-25  S3-fsdp-probe  探针实测（GPU 5,6 world_size=2，FULL_SHARD + use_orig_params，diffsynth_sf torch 2.11）：gen-phase backward 后 **idle(critic) orig-param grad = `None`**、AdamW step 计数 `[]` → **问题② 在 torch 2.11 不成立**（FSDP `_use_sharded_grad_views` 用 `_is_grad_none_mask` 正确把"无梯度的 idle 参数"标 None，AdamW 本就跳过；之前担心的 grad=0 未出现）。⇒ generator 有效 lr **没有**被 2.4× 抬高，weight-decay 也没多施加。`_none_out_grads` 降级为**前向兼容 safety net**（2.11 下是 no-op，防未来 torch 版本改 grad-view 行为）。**问题①（grad-clip 传子集→单-rank 局部范数）仍为真**（已从 accelerate 源码确认），`model.parameters()` 全集修复有效且必要。探针未打断 danze 的 GPU4-7 作业（~1GB/卡、~10s、独立端口 29701）。✅
- 2026-05-25  doc  补 Stage 3 "FSDP 存点 / Resume 拓扑语义"：模型+优化器走 `SHARDED_STATE_DICT`(DCP) **可 reshard**（8→4 成功根因）；EMA fp32 shadow 按 rank 存 `ema-rank{N}.safetensors`(非 DCP) **不可 reshard** → 改卡数即 EMA 重置（只丢平均历史、不丢已导出快照，2026-05-24 8→4 已实测）。操作建议=resume 保持 world_size 一致;review §4.1 加交叉引用。✅
