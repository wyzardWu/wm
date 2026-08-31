# ReactiveGWM_casual_forcing — 设计方案 (PLAN)

> 状态 (2026-06-06)：**Stage 1/2/3 均已训练并验证通过**（`sf3_casual_forcing_2` lineage）；Stage 3 长程 rollout 延伸 run 仍在进行中。run-by-run 进度见 [implement.md](./implement.md)，复现见 [README.md](./README.md)。
> 语言：本文档与后续交流均用中文。
> **唯一算法来源**：[thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)（含 Causal Forcing 与 Causal Forcing++ 三阶段）。
> **与 `examples/ReactiveGWM_self_forcing/` 完全隔离**：CF 的 model / pipeline / training module / runner / 推理代码**全部自己重新写**，不参考、不复用、不引用 SF 项目任何代码或符号；仅借用本仓库已有的**双向 base** (`ReactiveGWMModel` + `ReactiveGWMPipeline`) 作为权重起点与 Stage 3 DMD 的 teacher / critic。

把现有的 **双向 SF3 游戏世界模型**（`ReactiveGWM_base.safetensors`）通过 **Causal Forcing++ 三阶段蒸馏**，改造成 **4 步 / 逐帧自回归、KV-cache 实时**的因果生成器。

核心原则：

1. **唯一算法参考是 CF 上游**：Stage 1 AR Diffusion Teacher Forcing、Stage 2 Causal Consistency Distillation、Stage 3 Asymmetric DMD（CF 论文 v2）三段全部按上游实现。
2. **完全隔离 SF 项目**：不参考 `examples/ReactiveGWM_self_forcing/`，不引用 `diffsynth/models/reactive_gwm_self_forcing_dit.py` 与 `diffsynth/pipelines/reactive_gwm_self_forcing.py`；本项目 `diffsynth/` 新增的 CF model / pipeline 与 SF 同级文件并列存在但互不引用。
3. **CF 的 `CausalForcingReactiveGWMModel` 直接继承双向 `ReactiveGWMModel`**（不继承 SF 的 `CausalReactiveGWMModel`），自带 causal attention + KV-cache + TF dual-block，移植自 CF 上游 `wan/modules/causal_model.py`。
4. **训练框架对齐 DiffSynth accelerate**（与 `examples/ReactiveGWM` 一致）：`DiffusionTrainingModule.forward(data) -> loss`，Stage 1/2 单优化器 + Stage 3 双优化器 + EMA。
5. **首帧 anchor**：全三阶段 frame 0 始终为 GT latent，loss 不算 frame 0；推理走双向 base 同款 I2V 路径。

---

## 0. 已确认的关键决策（来自对话与批注）

| # | 决策 | 选择 |
|---|---|---|
| 1 | Stage 2 形式 | **仅 Causal CD（Causal Forcing++）**：不实现离线 ODE 配对 |
| 2 | Stage 3 实现 | **自包含 asymmetric DMD**：参考 CF 上游 v2 DMD（`model/dmd.py`、`pipeline/self_forcing_training.py`、`model/base.py`、`trainer/distillation.py`、`configs/causal_forcing_dmd_framewise.yaml`），自己写，不复制 SF（上游真实文件见 [stage3_cf_alignment_review.md](./stage3_cf_alignment_review.md) §1） |
| 3 | Stage 1 起点 | 自包含写新版 AR-TF，从双向 `ReactiveGWM_base` 起步 |
| 4 | Stage 1 损失 | **纯 teacher forcing**：dual-block `[clean ‖ noisy]` + 对齐上游 TF mask |
| 5 | TF 实现 | **A. 双块 `[clean ‖ noisy]` + 特殊 TF mask**（CF 上游 `_prepare_teacher_forcing_mask`） |
| 6 | 输出步数变体 | **仅 4-step**：`denoising_step_list=[1000, 750, 500, 250]` |
| 7 | 首帧条件化 | **锁定第 0 帧为 clean anchor**：三阶段训练 frame 0 都是 GT latent |
| 8 | 训练步数 | **三阶段统一 `max_steps=50000` 上限**（CF++ 论文 = Stage1/2/3 各 **20K / 5K / 1K @ bs64**；本项目 bs 4-8 小 8-16×，统一放大到 50K 上限；**Stage 3 因 DMD 易崩明确靠早停兜底**，见 §2 Stage 3 论文对照）；bs=8（8 卡 × 1 或 4 卡 × grad_accum 2） |
| 9 | 并行 | **DDP 优先，OOM 再切 FSDP**（与本仓库 SFT / distill 现有兜底一致；单 unit） |
| 10 | **SF 项目隔离** | CF 的 model / pipeline / module / runner / 推理**全部自己重新写**；不参考、不复用 SF；将来两边各自演进、不互影响 |

固定输入路径：

- 双向 base（三阶段所有 student 都从它初始化；Stage 3 teacher / critic 也是它）：
  `/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Models/SF3/ReactiveGWM_base.safetensors`（≈10 GB，`ReactiveGWMModel` DiT state dict）
- 数据：
  `/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF3/metadata.csv`
  列：`video,action,prompt`；`video=clips/clip_xxx/video.mp4`，`action=clips/clip_xxx/actions.parquet`（10 按键列），`prompt`=逐 clip NPC 行为描述
- Wan2.2-TI2V-5B 基座：
  `/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B/`
  tokenizer：`/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl`
- SF3 profile（来自 `examples/ReactiveGWM/data/profiles.py`）：480×832，`num_frames=101`，10 按键，`action_hold_window=10`

---

## 1. 参考来源与本地代码关系

### 1.1 算法参考（**唯一**）

| 来源 | 重点参考内容 | 用途 |
|---|---|---|
| `https://github.com/thu-ml/Causal-Forcing` | • `wan/modules/causal_model.py`（causal attention + KV-cache + `_prepare_teacher_forcing_mask`）<br>• `utils/wan_wrapper.py::WanDiffusionWrapper.forward`（`clean_x` 分支）<br>• `trainer/diffusion.py`、`pipeline/teacher_forcing_training.py`（Stage 1 AR-TF）<br>• `model/naive_consistency.py`、`trainer/naive_cd.py`（Stage 2 Causal CD）<br>• `model/dmd.py`（DMD KL grad / generator & critic loss）<br>• `pipeline/self_forcing_training.py`（4-step self-rollout / backward simulation）+ `model/base.py`（`_run_generator`：帧数采样 / 窗口 slice / gradient mask）+ `trainer/distillation.py`（双优化器 / 交替 / EMA / save）（Stage 3 DMD）<br>• `pipeline/causal_inference.py`（4-step KV-cache 推理）<br>• `configs/{ar_diffusion_tf,causal_cd,causal_forcing_dmd}_framewise.yaml` | 全部三阶段（AR-TF / CD / DMD）+ 推理 |

### 1.2 本地工程参考（仅范本，不复制）

| 模块 | 作用 |
|---|---|
| `examples/ReactiveGWM` | 双向 SFT (`ReactiveGWMTrainingModule` + accelerate + 双向 `ReactiveGWMPipeline` / `ReactiveGWMModel`)；**训练框架与数据范本**；直接复用 `data/`（profile / parquet→按键 / CSV prompt） |
| `diffsynth/models/reactive_gwm_dit.py` | 双向 `ReactiveGWMModel`、按键 embedder、`prepare_action_binned` / `inject_at_block`；**CF student 的父类** + Stage 3 teacher / critic 来源 |
| `diffsynth/pipelines/reactive_gwm.py` | 双向 `ReactiveGWMPipeline`、Wan VAE / T5、`model_fn_wan_video`；Stage 3 teacher / critic holistic score 路径 |
| `diffsynth/diffusion/*` | `DiffusionTrainingModule`、`ModelLogger`、`FlowMatchScheduler`、`runner.py`（基础训练循环） |

### 1.3 明确不采用 / 不参考

- **不参考 `examples/ReactiveGWM_self_forcing/`**（批注：里面有点问题；要求 CF 与 SF 完全隔离）。
- **不引用 / 不继承 `diffsynth/models/reactive_gwm_self_forcing_dit.py` 与 `diffsynth/pipelines/reactive_gwm_self_forcing.py`**；本项目新增的 `reactive_gwm_casual_forcing_dit.py` 与 `reactive_gwm_casual_forcing.py` 自包含、与 SF 文件并列但互不引用。
- 不参考 `https://github.com/guandeh17/Self-Forcing`、`https://github.com/tianweiy/CausVid`。
- 不参考 `examples/ReactiveGWM_ar/`、`examples/ReactiveGWM_distill/`。
- 不实现离线 ODE 配对（决策 1：仅 CD）。
- 不生成 1-step / 2-step 变体（决策 6：仅 4-step）。

---

## 2. Causal Forcing++ 三阶段配方（落到本代码库的形态）

三个角色（generator / teacher / critic）在不同阶段身份会变：

| 阶段 | generator（student） | teacher（真实分数） | critic（伪分数） |
|---|---|---|---|
| Stage 1 AR-TF | `CausalForcingReactiveGWMModel`（**新**子类，从双向 base 起步），**可训练**，multi-step AR | — | — |
| Stage 2 CD | 同 Stage 1 模型，从 Stage 1 产物起步，**可训练**；维护 EMA-student 副本 | **Stage 1 AR teacher 冻结副本**（同 `CausalForcingReactiveGWMModel`，TF 路径，做单步 ODE） | **目标网络**=EMA-of-student（同结构，`requires_grad=False`；**非 critic**，CD 无真假分数） |
| Stage 3 DMD | 同 Stage 1/2 模型，从 Stage 2 产物起步，**可训练**，4-step 自 rollout（**关闭** TF 路径，走 KV-cache） | **冻结的双向** `ReactiveGWMModel` | **可训练的双向** `ReactiveGWMModel` 副本 |

> 关键不变量：generator 跨三阶段共用同一个 `CausalForcingReactiveGWMModel`；state dict 与双向 base 100% 兼容。**训练路径**：Stage 1 / 2 走 TF 双块路径（带 `clean_x` 入参），Stage 3 走 KV-cache self-rollout（关闭 TF）。**推理路径**（sanity / 最终）：**三阶段统一走 KV-cache 自回归**，对齐 CF 上游 `pipeline/causal_diffusion_inference.py`。所以 `CausalForcingReactiveGWMModel` 从 Stage 1 起就要同时实现 TF dual-block 和 KV-cache 两条路径。三个 ckpt 都是 bare `CausalForcingReactiveGWMModel`，导出即下一阶段 `student_init`。

### Stage 1 — AR Diffusion Teacher Forcing（无 critic）

对齐 CF 上游 `trainer/diffusion.py` + `pipeline/teacher_forcing_training.py`。

- **角色**：generator（可训练）+ text encoder（冻结）+ VAE（冻结，仅 encode）。
- **输入**：SF3 clip → `num_frames=101` 像素帧 → `(101-1)//4+1 = 26` 个 latent 帧（Wan2.2 VAE 时序压缩 4×）；frame 0 固定为 GT clean anchor（决策 7，I2V）。
- **TF dual-block 构造**（移植 CF 上游 `_prepare_teacher_forcing_mask`）：
  - `clean_tokens = patchify(GT_latents)`、`noisy_tokens = patchify(GT_latents + ε_t)`。
  - 拼接 `[clean_tokens ‖ noisy_tokens]`，长 `2·26·frame_seqlen`。
  - flex_attention block mask：
    - `q_clean_i` attend `k_clean_{≤i}`（clean 半完全 causal）；
    - `q_noisy_i` attend `k_clean_{≤i-1} ∪ k_noisy_i`（之前的 clean 上下文 + 同 block 内 noisy tokens）；
    - clean→noisy 全屏蔽；noisy→未来 clean / 未来 noisy 全屏蔽。
  - timestep：clean 半 `t=0`，noisy 半 `t=t`；走 `seperated_timestep=True` 逐帧 timestep modulation。
- **采样调度**：**逐帧独立 timestep**（CF 上游 `uniform_timestep=False`，`model/diffusion.py::_get_timestep(0, 1000, uniform_timestep=False)` + `base.py:87-110`），`FlowMatchScheduler(shift=5.0, sigma_min=0.0)` 1000 步离散，全 `[0, 1000)` 范围逐帧采样（**不截断**）；每帧拿到独立噪声水平 = AR diffusion 的 data efficiency 来源。
- **损失**：仅在 noisy 半 26 帧上算 `MSE(flow_pred, noise - x0)` 并乘 **per-frame `training_weight(t)`**（BSMNTW 钟形权重，对齐 CF 上游 `model/diffusion.py:120-124`）；frame 0（anchor，I2V）loss mask = 0；clean 半预测**完全丢弃**。
- **优化**：单 AdamW（`lr=2e-6`、`betas=(0.0, 0.999)`、`wd=0.01`、`max_grad_norm=10`；对齐 CF 上游 `ar_diffusion_tf_framewise.yaml` + `default_config.yaml`）；bf16 + gradient checkpointing；DDP-first。
- **产物**：`stage1_ar.safetensors`（bare `CausalForcingReactiveGWMModel`）。

### Stage 2 — Causal Consistency Distillation（Causal Forcing++）

对齐 CF 上游 `model/naive_consistency.py::generator_loss` + `trainer/naive_cd.py` + `configs/causal_cd_framewise.yaml`。

- **角色（3 个网络，均从 Stage 1 产物初始化）**：generator/student（可训练）+ EMA-student（**目标网络** target network，`requires_grad=False`，EMA-of-student）+ teacher（**Stage 1 冻结副本**，`requires_grad=False`）+ 冻结 text encoder + VAE。注：EMA-student 是 CD 的"目标网络"，**不是** DMD 的 critic（CD 无 critic / 真假分数概念）。
- **action 条件（项目适配，上游 Wan 无）**：teacher / student / EMA 三方 forward 都带**同一份 action**（来自该 clip），随 TF `clean_x` 路径经 per-block embedder 注入；CD 一致性 = "给定同一 prompt + action 序列"下的 x0 一致性。
- **timestep：uniform（单 t broadcast 到所有帧）**——与 Stage 1 的逐帧独立**相反**。CD 沿单条全局 ODE 轨迹（一个 t 参数）训一致性，整段视频同噪声水平一起去噪。**实现时不要复用 Stage 1 的逐帧 `randint`。**
- **CD 调度**：`FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)`，`set_timesteps(discrete_cd_N=48)` → **48 个 timestep 节点**（index 0..47）。`extra_one_step` = `linspace(sigma_start, sigma_min, 48+1)[:-1]` = 48 个点（**丢掉 σ=0 端点**，避开 t=0 退化 / `1/σ` 奇异；**是 48 不是 49**）。diffsynth `FlowMatchScheduler("Wan").set_timesteps(48)` 本身就是此行为，天然对齐。
- **每个训练 step**：
  1. 数据集取 GT clean latent `x0`（26 帧）；frame 0 为 I2V anchor。
  2. 随机 `idx ∈ [0, N-1) = [0, 47)`：`t = timesteps[idx]`、`t_next = timesteps[idx+1]`，各 broadcast 到全 26 帧。
  3. 加噪 `latent_t = (1-σ_t)·x0 + σ_t·ε`（全帧同 σ_t）。frame 0 anchor：`latent_t[:,0]=x0[:,0]`、`t[:,0]=0`。
  4. **teacher 单步 ODE 推进**（`no_grad`，TF 双块路径 `clean_x=x0`）：**negative-prompt CFG（必须，`guidance_scale=3.0`）** `v = v_uncond + g·(v_cond − v_uncond)`（`v_*` 取 forward **第 1 个返回值 = flow**）；`latent_t_next = latent_t − (t − t_next)/1000 · v`。frame 0 anchor：`latent_t_next[:,0]=x0[:,0]`、`t_next[:,0]=0`。
  5. **student**（保留梯度，TF 路径 `clean_x=x0`）：forward 得 `flow_pred`，转 **x0 预测** `cm_pred_t = latent_t − σ·flow_pred`（σ 用与加噪一致的 **per-frame sigma**，frame0=0）。注意：上游 wrapper 直接返回 `(flow, x0)` 取第 2 个；**我们的 `forward` / `model_fn` 只返回 flow，所以这步转换在 CD helper 里做——不改默认返回类型**（见 implement.md 保护规则）。关键：TF mask off-by-one——noisy 帧 i 只能看 `clean_{<i}`（严格之前）+ `noisy_i` 自己，**看不到 clean 帧 i**，所以 `cm_pred_t` 是整段 26 帧干净视频的**真去噪估计** `[B,26,C,h,w]`，不是抄 `x0`。
  6. **EMA-student**（`no_grad`，TF 路径 `clean_x=x0`）：同样 `cm_pred_t_next = latent_t_next − σ_next·flow_pred` 在 `(latent_t_next, t_next)`（σ_next=`sigmas[idx+1]`，frame0=0）。
  7. **CD loss**：`loss = MSE(cm_pred_t[:,1:], cm_pred_t_next[:,1:])`——纯 MSE，**无 `training_weight`**（与 Stage 1 不同）；frame 0（I2V anchor）排除。
- **一致性原理**：`latent_t`、`latent_t_next` 是同一段视频在同一条 ODE 轨迹上的两点（teacher 一步连接）；两者的 x0 估计应相等 → 强制 `f(latent_t,t)=f(latent_t_next,t_next)` 对所有相邻节点成立 → 链式传递 → `f(任意噪声)=x0` → few-step 能力。**TF / 动作能力**靠 ①Stage 1 初始化 ②teacher=冻结 Stage 1（ODE 步锚住其动力学）③`clean_x` 上下文 继承，**不靠 GT 回归**。EMA 当 target 稳住自指训练（防 collapse）。
- **优化**：单 AdamW（`lr=2e-6`、`betas=(0.0, 0.999)`、`wd=0.01`、`max_grad_norm=10`，与 Stage 1 已对齐）；**EMA `decay=0.99` 从 step 0 就更新**（`ema_start_step=200` 仅决定"存 EMA 还是 raw"，**非**"200 步才开 EMA"）；bf16；DDP-first（3×5B 同驻，见 §9）。
- **产物**：`stage2_cd.safetensors`（**优先导出 EMA 权重**，bare `CausalForcingReactiveGWMModel`）。

### Stage 3 — Asymmetric DMD self-forcing（CF 上游 v2 DMD，自包含实现）

对齐 CF 上游 `model/dmd.py`（`class DMD`：KL grad / generator loss / critic loss）+ `pipeline/self_forcing_training.py`（4-step self-rollout / backward simulation）+ `model/base.py`（`SelfForcingModel._run_generator`：帧数采样 / 窗口 slice / gradient mask）+ `trainer/distillation.py`（双优化器 / 交替更新 / EMA / save）+ `configs/causal_forcing_dmd_framewise.yaml`。**自己写**，不复制 SF。

> 上游真实文件、机制偏离与决议均按 [stage3_cf_alignment_review.md](./stage3_cf_alignment_review.md) 订正（原 PLAN 引用的 `pipeline/causal_forcing_training.py` / `utils/dmd_loss.py` 上游**不存在**）。下方区分「上游默认行为」与「**项目适配**」（review §3.1 决议：迁移项目命名不必同上游，但行为要标注清楚）。

> **CF++ 论文实现细节对照（2026-05-22）**——论文方法与本设计一致，仅规模 / 尺寸有项目偏离：
> - **4-step schedule 已对齐**：论文 4-step `t=[1, 0.9375, 0.8333, 0.625]` = 本项目 `denoising_step_list=[1000,750,500,250]` 经 `warp_denoising_step + shift=5.0` 的 warped 值（`σ'=5σ/(1+4σ)` 逐点等价，已验证）。**仅 4-step**（决策 6）；论文的 2/1-step + ASD「首帧仍 4 步、后续帧降到 2/1 步」trick **out-of-scope，不实现**。
> - **score model 尺寸偏离**：论文 teacher / critic = Wan2.1-**14B**（generator 是 1.3B → 含尺寸非对称）；本项目只有 **5B 双向 base**，teacher / critic 同为 5B → 我们的「asymmetric」仅指**结构非对称**（双向 score vs 因果 generator），**无尺寸非对称**（无 14B 可用，硬约束）。
> - **步数 / batch（决策 2026-05-22：保持 50K 上限 + 早停）**：论文 Stage 3 = **1K 步 @ batch 64**；本项目 batch 4-8（小 8-16×）。按样本曝光，论文 1K×64=64K 视图 ≈ 本项目 batch8 的 8K 步 / batch4 的 16K 步。**仍保留统一 `max_steps=50000` 上限，但 Stage 3 明确靠「每 2500 step sanity 盯崩溃 / plateau → 早停」兜底**（DMD self-rollout 易崩，50K 是 cap 不是必跑完；曝光达论文 3-6×，过训风险用早停规避）。其余超参同 CF 上游。

- **角色**：generator（可训练，Stage 2 起步）+ EMA-generator（**仅用于导出**，不进任何 loss）+ teacher（**双向** `ReactiveGWMModel` 冻结）+ critic（**双向** `ReactiveGWMModel` 可训练副本）+ text encoder + VAE。
- **Rollout（KV-cache self-forcing，关闭 TF dual-block，`clean_x=None`）**：4-step `denoising_step_list=[1000,750,500,250]`；**26 帧定长窗口**（frame0=数据集 GT 干净 anchor，**不进 loss** + 25 生成帧；review §3.5 决议，取代上游 `num_training_frames=20`）。
  - **训练 rollout 用固定全量 buffer**：`kv_cache_size = 26 · frame_seqlen`，**训练阶段不触发 sink+recent 驱逐**（对齐上游 `pipeline/self_forcing_training.py:46,288-302`；review §3.3 决议）。sink+recent 驱逐是**推理**长视频才用的机制，默认 `sink_size=0`（兼容、日后可设 >0 开启）。
  - **梯度策略 = shared random exit step（`same_step_across_blocks=true`）+ 窗口梯度**（上游 framewise 行为，**不是** `grad_horizon=8`）：每次 rollout 随机采一个 exit step；因 `same_step_across_blocks=true`，所有 block 共享该 exit step（不是每 block 独立采样）；**只有 exit-step 那一次 forward 进计算图**，中间 step 与 context-cache 更新全 `no_grad`；帧维上只有 DMD 窗口（上游 last-21、本项目 26 帧除 frame0）保留梯度（上游 `start_gradient_frame_index`，`pipeline/self_forcing_training.py:141,170-173,182-228`；review §3.4）。
  - 首帧 anchor latent 由数据集提供，先 encode 进 KV-cache，rollout 从 frame 1 开始（**项目适配**：上游用「解码末帧→重编码」造 image-latent anchor，本项目 I2V 直接用 GT 干净首帧取代，review §3.4(3)/§3.5）。
- **DMD score（x0 空间，review §4.5；上游 `model/dmd.py:56-128,130-197`）**：teacher / critic 一次性 score 整段 26 帧 trajectory，**单 uniform timestep**（上游行为；本项目沿用「holistic」叫法——上游无此 flag 名，属 review §3.2 的「项目名=上游默认行为」）。
  - 必须 **flow→x0 转换**：`real_x0 = xt − σ·real_flow`、`fake_x0 = xt − σ·fake_flow`（teacher/critic forward 默认返回 flow，**不改默认返回类型**，与 Stage 1/2 一致）；
  - `grad = fake_x0 − real_x0`；`normalizer = mean(abs(generated_x0 − real_x0))` 沿 `[1,2,3,4]` 维、无 clamp、最后 `nan_to_num`；`loss_gen = 0.5·MSE(generated_x0, (generated_x0 − grad).detach())`（double 精度）。
  - CFG：`real_guidance_scale=3.0`、`fake_guidance_scale=0.0`（仅 cond fake score）。
  - timestep：uniform 采样 → `timestep_shift=5.0` → **clamp `[20,980]`**（review §4.3）；`ts_schedule=false`（review §4.4）。
- **critic loss**：critic 把 x0 预测转 flow 预测后算标准 flow-matching denoising loss（`denoising_loss_type=flow`，`model/dmd.py:296-328`）。**上游是整窗一次 backward**（`trainer/distillation.py:293`）；本项目若为显存引入 `per_frame_backward` / `max_dmd_frames` 属**项目适配、待显存实测后拍板**（review §3.6，非上游行为）。
- **交替更新（上游 `trainer/distillation.py:303-336`）**：`dfake_gen_update_ratio=5`——generator 每 5 步训一次、critic 每步训一次；双优化器：`lr=2e-6`（generator，`beta1=0.0`/`beta2=0.999`）/ `lr_critic=4e-7`（critic，`beta1_critic=0.0`/`beta2_critic=0.999`）；grad clip 10。
- **EMA（仅导出，review §4.1）**：`decay=0.99`，**上游式懒创建**——`step<ema_start_step=200` 不跟踪，到 200 才建；`update()` **只在 generator 优化器 step 之后**（即每 5 步）调用，critic step 不碰 EMA。与 Stage 2「in-loss 目标网络、step 0 起」不同是因为**用途不同**（Stage 3 EMA 不进 loss），二者各自独立、互不影响 Stage 1/2。
- **产物**：`stage3_dmd_generator.safetensors`、`stage3_dmd_ema.safetensors`（**最终导出**）、`stage3_dmd_critic.safetensors`。

---

## 3. 适配 DiffSynth accelerate 训练框架

DiffSynth 约定：`Accelerator` → `DiffusionTrainingModule.forward(data) -> scalar_loss` → runner `backward / step / save`。

- Stage 1 / 2 单优化器、标准 forward → 套基础 runner。
- Stage 3 需要双优化器、交替更新、EMA、多步 self-rollout、3×5B 模型同驻 → 自己写 `dmd.py` 训练 module + 自己写 DMD runner，参考 CF 上游 `trainer/distillation.py`（双优化器交替 / EMA / save）+ `pipeline/self_forcing_training.py`（rollout）+ `model/dmd.py`（loss）。

适配步骤：

1. **`diffsynth/` 新增 CF 专用 model / pipeline（自包含，与 SF 完全独立）**：
   - `diffsynth/models/reactive_gwm_casual_forcing_dit.py`：`CausalForcingReactiveGWMModel(ReactiveGWMModel)`——直接继承**双向** `ReactiveGWMModel`，自带：
     - causal block-attention（移植 CF 上游 `wan/modules/causal_model.py`）；
     - per-block KV-cache（含 sink+recent 驱逐，移植 CF 上游）；
     - dual-block TF 模式（`_prepare_teacher_forcing_mask` + `clean_x` 入参分支）；
     - `seperated_timestep=True` 逐帧 timestep modulation；
     - state dict 与双向 base / 父类 100% 兼容（不新增 `nn.Parameter`）。
   - `diffsynth/pipelines/reactive_gwm_casual_forcing.py`：自包含——
     - `model_fn_causal_forcing(model, ..., tf_mode: bool, clean_x=None, kv_cache=None)`：TF / KV-cache 两条路径分发；
     - CD step helper（teacher 单步 ODE + negative-prompt CFG）+ **flow→x0 转换**（`cm_pred = latent − σ·flow`，σ=加噪 per-frame sigma；CD 比 x0 预测不是 flow；**保持 `model_fn` 默认返回 flow**，转换在 helper 内或加 `return_x0` 可选参数）、holistic DMD helpers（**x0 空间**：`x0=latent−σ·flow`；teacher CFG score `g=3` / critic x0→flow / normalizer `mean(abs(generated_x0−real_x0))` 无 clamp+`nan_to_num` / `loss_gen=0.5·MSE(generated_x0,(generated_x0−grad).detach())`；review §4.5）；
     - VAE / T5 取自 `ReactiveGWMPipeline.from_pretrained`；
     - **不引用 `reactive_gwm_self_forcing.py` 任何符号**。
   - 仅在 `diffsynth/{models,pipelines}/__init__.py` 加导出（如需要）。

2. **三个 stage 各自一个 `DiffusionTrainingModule`**：
   - `modules/ar_tf.py::CFARTrainingModule` —— Stage 1：generator + 冻结 text/VAE；`forward(data)` 返回 dual-block TF flow-match MSE。
   - `modules/cd.py::CFCDTrainingModule` —— Stage 2：generator + EMA-student（目标网络）+ 冻结 teacher（Stage 1 副本）+ 冻结 encoders + **negative-prompt uncond 编码**；`forward(data)` 返回 CD MSE（uniform t、x0 一致性、teacher CFG ODE、frame-0 anchor 排除）；module 内维护 EMA（从 step 0 更新）。
   - `modules/dmd.py::CFDMDTrainingModule` —— Stage 3：generator + EMA + 双向 teacher（冻结）+ 双向 critic（可训练）+ 冻结 encoders；`forward(data, phase)` 按 `phase in {'generator','critic'}` 返回对应 loss。

3. **runner 与入口**：
   - 单一 `train.py`：argparse + yaml merge + `--stage {ar_tf,cd,dmd}` 派发。
   - Stage 1 / 2 共用基础单优化器训练循环（写在 `train.py` 中，对齐 `examples/ReactiveGWM` runner 风格）。
   - Stage 3 独立双优化器交替循环（`modules/runner_dmd.py`，**从零写**，参考 CF 上游 `trainer/distillation.py` + `pipeline/self_forcing_training.py`）。

4. **数据**：复用 `examples/ReactiveGWM/data/`（`profiles.get_profile("sf3")`、`action_utils.get_action_op`、`prompt_utils.resolve_prompt(..., use_csv_prompt=True, prompt_column="prompt")`）+ `diffsynth.core.UnifiedDataset`。Stage 1 / 2 需要 GT video latent + prompt + action；Stage 3 只需 prompt + action + 首帧 latent。

---

## 4. 目标目录结构（精简，参考 CF 上游布局）

```text
diffsynth/
├── models/
│   └── reactive_gwm_casual_forcing_dit.py     # CausalForcingReactiveGWMModel(ReactiveGWMModel)
│                                              # 自带 causal attention + KV-cache + TF dual-block；与 SF 文件无关
└── pipelines/
    └── reactive_gwm_casual_forcing.py         # model_fn(tf/kv-cache) + CD/DMD/调度 helpers；与 SF 文件无关

examples/ReactiveGWM_casual_forcing/
├── PLAN.md
├── implement.md                               # 实现进度；每完成一个 Phase ✅ + 停下 review
├── README.md
├── __init__.py
├── train.py                                   # 入口：argparse + yaml + --stage 派发 + Stage 1/2 训练循环
├── modules/
│   ├── __init__.py
│   ├── ar_tf.py                               # Stage 1：CFARTrainingModule + TF dual-block loss
│   ├── cd.py                                  # Stage 2：CFCDTrainingModule + EMA + CD loss
│   ├── dmd.py                                 # Stage 3：CFDMDTrainingModule + rollout + holistic DMD loss
│   ├── runner_dmd.py                          # Stage 3 双优化器交替循环（DDP/FSDP 切换）
│   ├── dataset.py                             # SF3 26-latent-frame 窗口数据集
│   └── ema.py                                 # 轻量 EMA（Stage 2/3 共用）
├── inference/
│   ├── __init__.py
│   ├── causal_inference.py                    # 4 步 KV-cache 推理（移植 CF 上游 pipeline/causal_inference.py）
│   ├── infer.py                               # 单 clip CLI
│   └── eval_action.py                         # 13 动作轴评测 + 区域 motion energy / MSE
├── configs/
│   ├── default.yaml
│   ├── stage1_ar.yaml
│   ├── stage2_cd.yaml
│   └── stage3_dmd.yaml
├── launch/
│   ├── stage1_ar.sh
│   ├── stage2_cd.sh
│   ├── stage3_dmd.sh
│   └── infer.sh
└── scripts/
    └── verify.py                              # CPU 子命令：scheduler / ar_tf / cd / dmd / rollout / smoke；gpu 子命令兜底
```

> 比原方案精简：`core/` 整层并入 `modules/`；scheduler 工具直接写在 `diffsynth/pipelines/reactive_gwm_casual_forcing.py`；`encoders.py` / `teacher.py` / `rollout.py` 不单独建（VAE/T5 从 `ReactiveGWMPipeline.from_pretrained` 取在 module `__init__` 里加载；teacher score / rollout 收纳进 `modules/dmd.py`）。整体结构对齐 CF 上游 `wan/ + trainer/ + pipeline/` 三层抽象的扁平化。
> `data/` 不复制，sys.path 复用 `examples/ReactiveGWM/data/`（SF3 按键 schema / prompt 与 base 一致）。

---

## 5. `CausalForcingReactiveGWMModel` 设计要点（新子类）

**继承自双向** `diffsynth.models.reactive_gwm_dit.ReactiveGWMModel`（**不**继承 SF 的 `CausalReactiveGWMModel`）。不新增 `nn.Parameter`，state dict 与双向 base 100% 一致——`ReactiveGWM_base.safetensors` strict 加载 0 missing / 0 unexpected。

类内新增能力（全部移植自 CF 上游 `wan/modules/causal_model.py`，与 SF 实现独立）：

1. **Causal block-attention**：
   - 默认走 block-causal 注意力（每 `num_frame_per_block=1` 帧一个 block，块内 dense，块间 causal）；
   - flex_attention block mask 按 `(F_lat, frame_seqlen, num_frame_per_block, device)` 四元组缓存，避免每 step 重建 `BlockMask`。

2. **KV-cache 自回归路径**（**Stage 1 起即实现**，与 TF dual-block 并列；Stage 1+ sanity 推理 / Stage 3 训练 self-rollout / Stage 4 最终推理统一复用此路径，对齐 CF 上游 `pipeline/causal_diffusion_inference.py`）：
   - per-block KV-cache + sink+recent eviction（**仅推理**长视频用；`kv_window_size=16`，**默认 `sink_size=0`**，可设 >0 开启 attention sink；review §3.3 决议）；**Stage 3 训练 rollout 用固定全量 buffer、不驱逐**；
   - cache-only refill：history 帧（含首帧 anchor）只写 K/V、不算 attention 输出；
   - cross-attn cache 一次写、持续读（prompt T5 encoder 编码一次后所有帧复用）。

3. **TF dual-block 模式（Stage 1/2）**：
   - forward 签名：`forward(noisy_x, timestep, ..., clean_x=None, kv_cache=None)`。
   - `clean_x is not None and kv_cache is None`：走 TF 路径——
     - 分别 patchify `clean_tokens` / `noisy_tokens`，拼成 `[clean ‖ noisy]` 总长 ×2；
     - timestep：`[zeros_like(t) ‖ t]` per-frame（走 `seperated_timestep=True`）；
     - flex_attention mask 调 `_prepare_teacher_forcing_mask`（按四元组缓存）；
     - 输出取 noisy 半 `[B, F_lat, C, H, W]`，clean 半完全丢弃。
   - `clean_x is None and kv_cache is not None`：走 KV-cache 自回归（**Stage 1+ sanity 推理 / Stage 3 训练 self-rollout / Stage 4 最终推理**）。
   - `clean_x is None and kv_cache is None`：走 plain block-causal（verify / 调试用）。

4. **首帧 anchor**（决策 7）：
   - frame 0 始终是 GT。Stage 1/2 TF 路径下 `noisy_x[:, 0] = clean_x[:, 0]` 不加噪 + frame 0 timestep=0；loss mask 在 frame 0 为 0。
   - Stage 3 rollout：首帧 latent 由数据集提供，先 encode 进 KV-cache，rollout 从 frame 1 开始。

5. **per-block 动作 graft**：
   - 30 个 `action_embedders.{i}.weight` 与父类 `ReactiveGWMModel` 完全一致；TF 路径下 `clean_x` 与 `noisy_x` 共享同一份动作条件，`inject_at_block` 同时注入到 clean 与 noisy token。

6. **RoPE / frame position**：
   - TF 双块下 clean 与 noisy 共享同一组 frame_position `[0,...,F_lat-1]`；mask 已隔离两半可见性，模型无须额外区分。与 CF 上游一致。

7. **cross-attn**：
   - clean 半与 noisy 半都接收 prompt cross-attn；KV-cache 模式下 cross-attn cache 由内部一次性写入并复用。

---

## 6. 数据与编码

- `modules/dataset.py`：
  - Stage 1 / 2：26 latent 帧连续窗口（101 像素帧），首帧 latent 为 anchor；输出 `{video_latents [B,26,C,h,w], action [B,T_pix,K], prompt_str}`；t 在 loss 内逐帧独立采样。
  - Stage 3：26 latent 帧的 prompt + action + 首帧 latent（决策 7：默认提供）。
- VAE / T5 编码：在每个 module `__init__` 里 `ReactiveGWMPipeline.from_pretrained(...)` 取 `vae` 与 `text_encoder`，冻结 + eval。
- 先不预编码 cache；后续如吞吐瓶颈再加 `scripts/precompute_cache.py`。

---

## 7. 配置与 launch 契约（8×H200）

`configs/default.yaml`：

```yaml
# 路径
base_ckpt:        /home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Models/SF3/ReactiveGWM_base.safetensors
wan_base_dir:     /home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B
tokenizer_dir:    /home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
metadata_path:    /home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF3/metadata.csv
dataset_base:     /home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF3
game: sf3
use_csv_prompt: true
prompt_column: prompt

# 几何（SF3 profile）
height: 480
width: 832
num_frames: 101
action_hold_window: 10
ode_latent_frames: 26          # Stage 1/2 窗口 (101 像素帧→26 latent)；Stage 3 = dmd_score_frames (=26)

# 4-step（决策 6）
denoising_step_list: [1000, 750, 500, 250]
num_train_timestep: 1000
timestep_shift: 5.0
num_frame_per_block: 1
warp_denoising_step: true

# Stage 1 AR-TF（对齐 CF 上游：逐帧独立 timestep + 全 [0,1000) 范围）
ar_uniform_timestep: false      # 逐帧独立（CF 上游 uniform_timestep=False）
# ar_min_step / ar_max_step 已弃用：上游 generator_loss 全 [0,1000) 范围, 不截断
ar_first_frame_anchor: true     # 决策 7（I2V）
ar_num_train_frames: 26

# Stage 2 CD（CF++）
cd_discrete_N: 48
cd_guidance_scale: 3.0
cd_ema_decay: 0.99
cd_ema_start_step: 200

# Stage 3 DMD（CF 上游 v2，自包含实现；上游真实文件 + 决议见 stage3_cf_alignment_review.md §1/§0.1）
# —— 字段名沿用本项目习惯（迁移项目，review §3.1）；注释标注「上游行为」or「项目适配」——
dmd_score_frames: 26            # DMD 窗口=26 帧（frame0 GT anchor 不算 loss + 25 生成）；上游用变长 rollout→last-21，本项目定长 26（review §3.5）
context_noise: 0
same_step_across_blocks: true   # 上游默认 True（False 分支 "Useless, never met" + 会断 DMD timestep 推导）（review §4.2）
last_step_only: false
stochastic_exit_step: true      # 项目名=上游行为：每次 rollout 随机 exit；same_step_across_blocks=true 时所有 block 共享该 exit（DMD2 backward sim）（review §3.4）
detach_history_kv: true         # 上游行为：中间 step + context-cache 更新走 no_grad（review §3.2/§3.4）
holistic_dmd: true              # 项目名 = 上游默认行为（整段单 uniform timestep score；上游无此 flag 名）（review §3.2）
independent_first_frame: false
ts_schedule: false              # 对齐上游 framewise yaml（review §4.4）
denoising_loss_type: flow       # critic x0→flow 后算 flow-matching denoising loss（review §4.5）
real_guidance_scale: 3.0
fake_guidance_scale: 0.0
dmd_timestep_clamp: [20, 980]   # DMD score / critic timestep：uniform → shift 5.0 → clamp（review §4.3）
dfake_gen_update_ratio: 5
beta1_critic: 0.0               # critic AdamW（review §6.6）
beta2_critic: 0.999
dmd_ema_decay: 0.99
dmd_ema_start_step: 200         # EMA 仅导出用：上游式懒创建（step<200 不跟踪）+ 只在 generator step 更新（review §4.1）
kv_window_size: 16              # 模型 KV-cache 滑窗（推理 / Stage 3 rollout）；Stage 1/2 模型 init 也读它（ar_tf.py/cd.py）。Stage 3 *训练* rollout 用全量 buffer 不驱逐（review §3.3）；sink_size 复用 sanity_sample_sink_size（已设 0）
dmd_train_kv_window: 26         # Stage 3 训练 rollout 全量 buffer，不驱逐
dmd_train_sink: 0
dmd_teacher_offload_cpu: true   # 项目显存适配：teacher 冻结且只在 generator DMD score 用；CPU offload 不改数学语义
dmd_empty_cache_each_phase: true # 项目显存适配：对齐上游定期 empty_cache/gc，降低 FSDP/rollout 碎片峰值
dmd_gc_interval: 100
sanity_sample_memory_fraction: 0.45 # Stage3 完整 9.4GB export + 4-step CFG sanity 需要高于默认 0.2
# exit-step checkpoint + KV-cache read-state snapshot：项目显存适配；只影响重算/显存，不改变 DMD2 rollout/loss。
# FSDP_IGNORED_MODULES="pipe.text_encoder|pipe.vae"：冻结 T5/VAE 排除 root flat-param，降低 all-gather 峰值。
# optimizer 实现：单 AdamW 两 param groups（gen/critic 各自 lr/betas），每 phase 单独 backward+step；等价保持上游交替更新语义，且 FSDP save_state 可保存 optimizer。
# —— 以下为「项目适配」候选（上游无对应物），待显存实测后定，引入务必标注非上游 ——
# max_context_frames: 7         # 上游无（项目 KV-window 适配？待定）
# grad_horizon: 8               # ✗ 上游无：梯度是 per-block exit-step + 窗口 mask，不是"最后 N 帧"（review §3.4）
# max_dmd_frames: 4             # 上游无：critic 整窗一次 backward（review §3.6）；当前 26 帧完整窗口已可训练，暂不需要
# critic_update_mode: per_frame_backward  # 上游无：同上；当前 26 帧完整窗口已可训练，暂不需要

# 优化
lr: 2.0e-6
lr_critic: 4.0e-7
beta1: 0.0                      # Stage 3 generator/critic
beta2: 0.999
weight_decay: 0.01             # CF 上游 default_config.yaml（Stage 1/2/3 共用）
ar_beta1: 0.0                   # Stage 1/2（CF 上游 beta1=0.0）
ar_beta2: 0.999
ar_grad_clip: 10.0

# 训练循环
seed: 42
dataset_repeat: 100
dataset_num_workers: 2
gradient_accumulation_steps: 1
mixed_precision: bf16
gradient_checkpointing: true

# Sanity 推理（三阶段统一走 KV-cache 自回归路径；每 save_steps spawn 非阻塞 subprocess，绝不阻塞训练）
sanity_sample_steps: 2500
sanity_sample_pixel_frames: 1500       # 60 秒 @ 25 FPS
sanity_sample_latent_frames: 300       # ≈ pixel_frames / VAE 时序压缩比
sanity_sample_diffusion_steps: 50      # Stage 1/2 multi-step；Stage 3 yaml 覆盖为 4
sanity_sample_kv_window_size: 16
sanity_sample_sink_size: 0             # review §3.3 决议：默认 0（纯 recent-window，各阶段兼容）；>0 开启 attention sink。Stage 1/2 历史 sanity 用 2（已验证）
sanity_sample_device: gpu_shared       # gpu_shared | cpu
sanity_sample_memory_fraction: 0.2     # gpu_shared 模式硬限 ~16 GB（5B bf16 + KV-cache ≈12 GB）
sanity_sample_clip_idx: 0              # fixed sanity 用第几个 clip 的 prompt + action
```

stage yaml 只覆盖差异项（**三阶段统一 `max_steps=50000`、`save_steps`、`sanity_sample_*`**，详见 [implement.md](./implement.md) 的"通用训练规格"表）。canonical 配置全部指向 `sf3_casual_forcing_2` lineage（精确字段以仓库内 `configs/*.yaml` 为准）：

- `stage1_ar.yaml`：`stage: ar_tf`，4 卡 `gradient_accumulation_steps: 2`（全局 batch 8），CF++ aligned 重训，`sanity_sample_diffusion_steps: 50`，`output_path: .../sf3_casual_forcing_2/stage1_ar`。
- `stage2_cd.yaml`：`stage: cd`，`student_init` = Stage 1 产物，`grad_accum: 1`（全局 batch 4），`sanity_sample_diffusion_steps: 50`，`output_path: .../sf3_casual_forcing_2/stage2_cd`。
- `stage3_dmd.yaml`：`stage: dmd`，`student_init` = Stage 2 产物，`sanity_sample_diffusion_steps: 4`（4-step 推理），`output_path: .../sf3_casual_forcing_2/stage3_dmd`。`stage3_dmd_long_from11200.yaml` 是长程 rollout 延伸变体。
- 旧 8card/4card + 一次性 ablation 配置已归档到 `configs/archive/`。

launch 脚本（`accelerate launch`，env 覆盖 `BASE_CKPT / DATA_ROOT / OUT / STUDENT_INIT / RESUME_STATE`；`NUM_PROCESSES` + `CUDA_VISIBLE_DEVICES` 切卡数；canonical 配置已是 launcher 默认 CFG）：

```bash
launch/stage1_ar.sh   # accelerate launch train.py --stage ar_tf --config configs/stage1_ar.yaml
launch/stage2_cd.sh   # 同上，stage=cd；env STUDENT_INIT=$(stage1 产物)
launch/stage3_dmd.sh  # 同上，stage=dmd；env STUDENT_INIT=$(stage2 产物)；--use_fsdp 兜底
launch/infer.sh       # 4 步 KV-cache 推理 + 13 动作轴评测
```

---

## 8. 推理（实时）

`inference/causal_inference.py`：4 步 KV-cache 推理；**移植自 CF 上游** `pipeline/causal_inference.py`，自包含写：

- 加载 `CausalForcingReactiveGWMModel` ckpt（`clean_x=None`，自动走 KV-cache 路径）；
- 首帧 GT latent → VAE encode → 写入 KV-cache（cache-only refill，不出 token）；
- 后续逐帧 4-step denoise `[1000,750,500,250]`；每生成一帧追加进 KV-cache，老帧按 sink+recent 驱逐；
- prompt 走 T5 一次性编码，cross-attn cache 复用；
- `inference/eval_action.py`：13 动作轴网格 + 区域 motion energy + MSE；评测 metric helper 可 importlib file-path 从 `examples/ReactiveGWM` 加载（避免包名冲突）。

v1 不把"实时数十 FPS"作为硬指标；先保证 4 步因果质量与双向 base 对齐度。

---

## 9. 8×H200 显存预算思路

- **DDP 优先，OOM 切 FSDP**（决策 9；单 unit，与本仓库 SFT / distill 现有兜底一致）。
- **Stage 1 / 2**：dual-block TF 把序列长度 ×2，激活也 ×2（flex_attention block mask 比 dense ×2 更紧）。预计 ~70-80 GB/GPU，DDP 多半够；OOM 立即切 FSDP。
- **Stage 3**：3×5B + 4 步 rollout + holistic DMD backward；实测需 FSDP NO_WRAP/FULL_SHARD + `mixed_precision: "no"` + teacher CPU offload + exit-step checkpoint。
- `ode_latent_frames=26` / `dmd_score_frames=26`（**不再用** `num_training_frames=20` / `grad_horizon=8`，见 review §3.4/§3.5）；训练 rollout 用固定全量 buffer（`26·frame_seqlen`），推理滑窗 `kv_window_size=16`。2026-05-23 已在 GPU0-3（邻居占 ~21GB/卡）完成 26 帧完整窗口 2-step memdiag。
- gradient checkpointing on（CF 上游 default）；Stage3 exit-step KV-cache forward 额外用 checkpoint + read-state snapshot，避免保存 25 帧整窗激活；teacher 冻结、`eval()`、`no_grad`，并默认 CPU offload；EMA（仅导出）只覆盖 `requires_grad=True` 参数子集。
- 梯度只走 per-block exit-step forward + DMD 窗口（中间 step/context 更新 no_grad），是上游天然的显存控制；`per_frame_backward` / `max_dmd_frames` 暂不需要（26 帧完整窗口已可训练），保留为极端 OOM 备选。

---

## 10. 风险与验证（`scripts/verify.py`，CPU + 单卡 GPU 子命令）

1. **state dict 兼容**：`CausalForcingReactiveGWMModel` 从 `ReactiveGWM_base.safetensors` strict 加载 0 missing / 0 unexpected；三阶段 ckpt 互相 strict 加载 0 missing。
2. **TF dual-block forward**：
   - mask 形状 `2·F·seqlen × 2·F·seqlen`；
   - `q_noisy_i` 只看 `k_clean_{≤i-1} ∪ k_noisy_i`；clean→noisy 全屏蔽；
   - clean 半完全 causal（`q_clean_i` 只看 `k_clean_{≤i}`）；
   - noisy 半 frame 0 anchor 模式下 loss mask = 0。
3. **TF vs KV-cache 一致性**：两条路径各自内部自洽即可（不要求互相数值一致——可见性不同）；分别 verify。
4. **调度一致性**：本地 FlowMatch 与 `diffsynth.FlowMatchScheduler("Wan")` 在 1000 步逐点一致；`cd_discrete_N=48` 离散 **48 节点**（`set_timesteps(48)` extra_one_step 丢 σ=0 端点）与 CF 上游 `causal_cd_framewise.yaml` 一致。
5. **AR-TF flow-match loss**：随机 GT latent + **逐帧独立 t** + dual-block forward → per-frame `training_weight` 加权 → loss 数值有限、梯度只到 generator、frame 0 anchor 模式下不进 loss。
6. **CD loss**（Stage 2）：
   - `cm_pred` 取 forward **第 2 个返回值 = x0 预测**（`latent − σ·flow`），**非 flow**；
   - student / EMA-student 同初始化时 `cm_pred_t ≈ cm_pred_t_next` → loss → 0；
   - teacher 单步 ODE 推进 = `latent_t − dt·v_cfg`，`dt=(t − t_next)/1000`（手算等价）；
   - negative-prompt CFG `v_uncond + 3·(v_cond − v_uncond)` 路径正确；
   - uniform t（全帧同 t）；frame-0 anchor 排除出 loss。
7. **Stage 3 DMD**（验证项按 review §6.8）：
   - self-rollout 4 步逐帧；**shared random exit step（`same_step_across_blocks=true`）**：每次 rollout 随机采一个 exit step，所有 block 共享；只有 exit-step forward 进计算图，中间 step + context-cache 更新 `no_grad`（`detach_history_kv`）；帧维只在 DMD 窗口（26 帧除 frame0）保留梯度（**非** `grad_horizon=8`）；训练 rollout 全量 buffer 不驱逐；
   - **DMD grad 在 x0 空间**：`x0=latent−σ·flow`，`grad=fake_x0−real_x0`，`normalizer=mean(abs(generated_x0−real_x0))` 无 clamp；
   - timestep uniform + `shift=5.0` + clamp `[20,980]`；`same_step_across_blocks=true`；
   - teacher CFG score / critic flow score 形状与 backbone 输出对齐；`loss_gen=0.5·MSE(generated_x0,(generated_x0−grad).detach())` 反传只到 generator；
   - **critic flow loss 与 DMD x0 loss 是两个不同空间**，critic loss 反传只到 critic；
   - frame0 anchor mask（frame0 不进 DMD loss）；
   - EMA 仅导出：懒创建（step<200 不跟踪）+ 只在 generator step 更新，只覆盖 `requires_grad=True` 参数；
   - DDP-safe：phase 切换时 `set_static_graph` / `find_unused_parameters` 行为可控。
8. **First-frame anchor**：frame 0 latent = GT；loss mask frame 0 = 0；推理时首帧 KV-cache 由 GT encode 填入。
9. **三阶段 ckpt 串联**：Stage 1 产物 → Stage 2 module strict 加载；Stage 2 EMA 产物 → Stage 3 module strict 加载；零权重丢失 / rename 错乱。
10. **GPU 子命令**：FA3 / SDPA vs flex_attention（TF mask 下）匹配；bf16 dual-block TF forward 数值有限 + 梯度通；遵守 HARD OPS RULE（避免与 8 卡训练同时占卡，必要时 `set_per_process_memory_fraction(0.05)` 兜底）。
11. **smoke test**（tiny CPU 端到端）：Stage 1 TF loss+backward / Stage 2 CD loss+backward（梯度只到 student）/ Stage 3 critic+generator loss+backward / inference shape 全过。

集群验证项：三阶段收敛曲线、4 步质量 vs 双向 base（`eval_action.py` 对照）。

---

## 11. 实现顺序（确认后执行；每个 Phase 完成后停下给 review，按 [`mark-progress-in-implement-md`](../../memory/mark-progress-in-implement-md.md) 在 `implement.md` 打 ✅）

Phase 1–6（骨架 → 数据/调度 → `CausalForcingReactiveGWMModel`（causal attn + KV-cache + TF dual-block）→ Stage 1 `ar_tf` → Stage 2 `cd`+`ema` → Stage 3 `dmd`+`runner_dmd`）**已实现**；Phase 7–8（推理评测 `inference/{causal_inference,infer,eval_action}.py` + verify 全套 + README）随 Stage 4 展开（Locked）。每步 self-contained，除 CF 专用文件外不动其他代码（**绝对不动** SF 项目文件）。

---

## 12. 已定 / 可调项

- `dmd_score_frames=26`（非 `num_training_frames=20` / `grad_horizon=8`，review §3.4/§3.5）；推理 `kv_window_size=16`、训练 rollout 全量 buffer。
- `per_frame_backward` / `max_dmd_frames`：**未采用**（26 帧整窗一次 backward 已可训，review §3.6）。
- `cd_discrete_N=48`（CF 上游 default，不收敛可试 32/64）；Stage 2 negative-prompt CFG `guidance_scale=3.0`（**必须**）。
- 输出目录 `.../sf3_casual_forcing_2/{stage1_ar,stage2_cd,stage3_dmd,stage3_dmd_long_from11200}`（canonical lineage；旧 `sf3_casual_forcing/` 谱系已被取代）。
- 后端：DDP 默认 → 实际 Stage 3 用 **FSDP NO_WRAP / FULL_SHARD**（见 implement.md §3.2.7 + FSDP 存点 caveat）。
- VAE+T5 预编码 cache：未接（保持简单，后续可加 `scripts/precompute_cache.py`）。
