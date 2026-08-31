# Rebuttal Training Plan

## 1. 目标与约束

在 SF2 数据集上设计三组 prompt / 参数训练消融实验。三个版本均：

- 从原始 Wan2.2-TI2V-5B 冷启动，不加载 `Vanilla.safetensors`；
- 使用视频首帧、目标视频和 action parquet；
- 保留并训练 ReactiveGWM 的 30 个 action embedder；
- 使用 480×608 分辨率、101 帧；
- 所有新增代码和训练入口只放在 `examples/Rebuttal/`；
- 不修改 `examples/ReactiveGWM/`、`diffsynth/`、源数据集或其他外部内容。

数据路径：

```text
DATA_ROOT=/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF2
VANILLA_METADATA=${DATA_ROOT}/metadata_vanilla.csv
STRUCTURED_METADATA=${DATA_ROOT}/metadata.csv
```

模型路径：

```text
WAN_ROOT=/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B
TOKENIZER=/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
```

## 2. 已验证的数据事实

- `metadata_vanilla.csv` 和 `metadata.csv` 均包含 10,000 个样本。
- 两份 CSV 的字段均为 `video,action,prompt`。
- 两表的 `(video, action)` 逐行一致；正式生成时仍按键连接，不依赖行号。
- `metadata.csv` 的每行 prompt 都包含且只包含一个结尾的 `Strategy(...)`。
- 数据中共有 9 种 Strategy：
  - Offense：2,603 条；
  - Control：2,956 条；
  - Defense：4,441 条。
- UMT5 prompt 长度上限为 512 tokens，实测：

| Prompt 类型 | 最大 tokens |
|---|---:|
| Vanilla | 100 |
| Vanilla + Strategy | 122 |
| 完整 structured prompt | 305 |
| Strategy only | 27 |

三个版本均不会触发 prompt 截断。

## 3. 实验矩阵

| 版本 | 初始化 | 训练 Prompt | 参数策略 |
|---|---|---|---|
| V1 `vanilla_strategy` | 原始 Wan2.2 | `<vanilla prompt> Strategy(...)` | 整个 DiT full fine-tune |
| V2 `strategy_only` | 原始 Wan2.2 | `Strategy(...)` | 整个 DiT full fine-tune |
| V3 `hybrid_cross_lora` | 原始 Wan2.2 | 原始完整 structured prompt | cross-attn LoRA，其余 DiT full fine-tune |

三版都继续读取：

```text
video
action parquet
input_image
```

V2 只删除 prompt 中的 `Active_Behavior(...)` 和 `Passive_Behavior(...)`，不会删除玩家 action 输入。

## 4. Prompt 构造

### 4.1 V1：Vanilla + Strategy

从 `metadata_vanilla.csv` 读取 vanilla prompt，从相同 `(video, action)` 对应的 `metadata.csv` prompt 中提取完整的 `Strategy(...)` 片段。

最终 prompt 使用一个 ASCII 空格连接：

```python
prompt = vanilla_prompt.rstrip() + " " + strategy_prompt
```

示例：

```text
Street Fighter 2, At the Air Force Base, ... Strategy(Defense: Holds ground with blocks and reactive counters, ...)
```

### 4.2 V2：Strategy Only

只保留结构化 prompt 末尾的完整 Strategy：

```text
Strategy(Defense: Holds ground with blocks and reactive counters, ...)
```

不添加 `NPC:` 前缀，也不保留 Active / Passive 文本。

### 4.3 V3：完整 Structured Prompt

直接使用 `metadata.csv` 的原始 `prompt` 列，不做文本变换。

## 5. 计划目录结构

```text
examples/Rebuttal/
├── plan.md
├── README.md
├── prepare_metadata.py
├── train.py
├── runner.py
├── checkpoint_io.py
├── eval.py
├── launch/
│   ├── _common.sh
│   ├── prepare_cache.sh
│   ├── train_v1_vanilla_strategy.sh
│   ├── train_v2_strategy_only.sh
│   └── train_v3_hybrid_cross_lora.sh
├── tests/
│   ├── test_metadata.py
│   ├── test_trainable_policy.py
│   └── test_checkpoint_io.py
└── generated/
    └── .gitignore
```

外部 ReactiveGWM 和 DiffSynth 文件只作为只读依赖导入。

## 6. 派生 Metadata

`prepare_metadata.py` 负责生成：

```text
examples/Rebuttal/generated/metadata_v1_vanilla_strategy.csv
examples/Rebuttal/generated/metadata_v2_strategy_only.csv
examples/Rebuttal/generated/metadata_manifest.json
```

生成流程：

1. 读取两份源 CSV。
2. 验证必需字段为 `video,action,prompt`。
3. 验证 `(video, action)` 在各自 CSV 内唯一。
4. 按 `(video, action)` 做一对一连接。
5. 使用严格的结尾规则提取 `Strategy(...)`。
6. 缺失、重复、格式异常或键不匹配时立即失败。
7. 分别生成 V1 和 V2 metadata。
8. 使用临时文件加原子替换，避免留下半成品。
9. 写入 manifest。

Manifest 记录：

- 两份源 CSV 的绝对路径及 SHA256；
- 输出 CSV 的 SHA256；
- prompt 构造模式和版本；
- 样本数量；
- 9 种 Strategy 的频次；
- 若干首尾样例；
- tokenizer 长度统计；
- 生成时间。

源 metadata 文件在生成前后进行 hash 校验，确保没有被修改。

## 7. 训练入口

`examples/Rebuttal/train.py` 复用现有 ReactiveGWM 模型、pipeline、action operator 和 loss，但在 Rebuttal 目录内独立实现：

- 三种 variant 配置；
- `--max_train_steps`；
- V3 混合可训练参数控制；
- 参数与 optimizer 审计；
- Rebuttal checkpoint 保存及恢复；
- cache manifest 与 metadata hash 校验。

三个 shell 入口只负责选择 variant 和提供路径，不复制核心训练逻辑。

启动前必须检查：

- 数据、Wan 权重和 tokenizer 文件存在；
- 派生 metadata 与 manifest 匹配；
- cache 与当前 metadata 匹配；
- `CUDA_VISIBLE_DEVICES` 非空；
- `--num_processes` 与可见 GPU 数量一致；
- 输出目录不会意外覆盖已有训练。

GPU 数量从 `CUDA_VISIBLE_DEVICES` 自动推导，正式训练使用 6 卡。

## 8. V1 / V2 参数策略

V1 和 V2 都从原始 Wan2.2 构建 `ReactiveGWMModel`：

- Wan 中名称和形状匹配的参数正常加载；
- 30 个 action embedder 使用 Xavier 初始化；
- 整个 DiT 设置为可训练；
- action embedder、self-attn、cross-attn、FFN、embedding、modulation 和 head 均参与 full fine-tune。

训练开始时输出：

- 可训练 tensor 数；
- 可训练参数总量；
- action / self-attn / cross-attn / FFN 各自的参数统计；
- 前若干个可训练参数名。

## 9. V3 混合训练策略

### 9.1 LoRA 目标

LoRA 仅注入：

```text
blocks.<0..29>.cross_attn.q
blocks.<0..29>.cross_attn.k
blocks.<0..29>.cross_attn.v
blocks.<0..29>.cross_attn.o
```

目标正则：

```text
.*\.cross_attn\.(q|k|v|o)
```

LoRA 配置：

```text
rank = 32
alpha = 32
```

预期：

```text
120 个 LoRA Linear
240 个 LoRA A/B tensor
23,592,960 个 LoRA 参数
```

### 9.2 冻结参数

以下参数冻结：

```text
blocks.*.cross_attn.q/k/v/o 原始 weight 和 bias
blocks.*.cross_attn.norm_q
blocks.*.cross_attn.norm_k
blocks.*.norm3
```

其中 `norm3` 虽然在代码结构中位于 `DiTBlock`，但它是 cross-attn 前的 LayerNorm，因此归入 cross-attn 分支并冻结。

### 9.3 Full Fine-tune 参数

除上述 cross-attn 原始分支以外，其余 DiT 参数全部 full fine-tune，包括：

```text
action_embedders
self_attn
ffn
norm1
norm2
patch_embedding
text_embedding
time_embedding
time_projection
modulation
head
其他所有非 cross-attn 参数
```

### 9.4 强制参数审计

训练启动前必须满足：

- LoRA 只能出现在 `cross_attn`；
- self-attn 不得出现 LoRA；
- cross-attn 原始 weight、bias、norm_q、norm_k 不可训练；
- `blocks.*.norm3` 不可训练；
- cross-attn LoRA A/B 可训练；
- action embedder、self-attn 和 FFN 可训练；
- LoRA tensor 和参数数量与 rank 32 的预期值一致；
- 所有可训练参数只能进入 optimizer 一次；
- 不存在遗漏的可训练参数。

任一断言失败时，训练直接退出。

## 10. 统一训练配置

三个版本使用同一套训练超参数：

```text
learning_rate              5e-5
weight_decay               0.01
gradient_accumulation      1
num_processes              6
effective_batch            6
max_train_steps            30000
save_steps                 1000
prompt_dropout_prob        0.1
action_dropout_prob        0.0
gradient_checkpointing     true
dataset_repeat             1
dataset_num_workers        4
height                     480
width                      608
num_frames                 101
action_hold_window         10
```

V3 的 LoRA 参数与其他 full-finetune 参数放入同一个 AdamW optimizer，统一使用 `5e-5`，不使用双学习率。

默认不启用 `accelerator.save_state`，避免保存巨大的 optimizer state；如后续确有无缝恢复需求，再通过显式参数开启。

## 11. Cache 计划

正确性验证阶段先使用 8～16 个样本的非缓存 metadata 做 smoke test。

正式训练使用 cached dataset：

- 三个版本共享 video VAE latent；
- 三个版本共享 first-frame VAE latent；
- V1、V2、V3 分别维护自己的 T5 cache 和 manifest；
- cache 启动入口位于 `examples/Rebuttal/launch/prepare_cache.sh`；
- cache manifest 必须记录并校验对应 metadata 的 SHA256；
- prompt mode、尺寸、帧数、tokenizer 和模型指纹不一致时立即失败。

为了避免重复保存三份相同 VAE cache，使用一个公共 VAE cache 目录，各 variant cache root 只保存独立 T5 cache和 manifest，并以明确的共享路径引用公共 VAE cache。

## 12. Checkpoint 设计

### 12.1 V1 / V2

每 1,000 step 保存标准完整 DiT：

```text
step-1000.safetensors
step-2000.safetensors
...
step-30000.safetensors
```

权重可以直接覆盖到由原始 Wan 构建的 `ReactiveGWMModel`。

### 12.2 V3

每个保存点输出：

```text
step-N.full.safetensors
step-N.lora.safetensors
step-N.manifest.json
```

内容：

- `step-N.full.safetensors`：所有非 cross-attn 的 full-finetune 参数；
- `step-N.lora.safetensors`：cross-attn LoRA A/B；
- `step-N.manifest.json`：原始 Wan 指纹、step、rank、alpha、目标正则、参数统计和文件 hash。

V3 的 weight-only 恢复顺序：

1. 从原始 Wan 构建 ReactiveGWM；
2. 注入相同配置的 cross-attn LoRA；
3. 加载 `step-N.full.safetensors`；
4. 加载 `step-N.lora.safetensors`；
5. 再次执行参数策略审计。

### 12.3 最终合并

提供最终导出功能，将：

```text
原始 Wan + full 参数 + fused LoRA
```

合成为标准完整 DiT：

```text
final_merged.safetensors
```

合并后执行逐 key、shape、dtype 检查，并用固定输入对比分离加载和 merged checkpoint 的推理结果。

### 12.4 存储预算

按照每 1,000 step 保存一次、训练 30,000 step：

- V1、V2 每个 checkpoint 接近一个完整 DiT；
- V3 保存非 cross-attn full 参数和较小的 LoRA；
- 三组训练合计预计需要约 0.75–0.85 TB checkpoint 空间；
- 该估算不包含 optimizer / accelerator state。

不自动删除旧 checkpoint，任何清理策略后续单独确认。

## 13. 验证流程

### 13.1 静态检查

- Python 语法检查；
- shell 语法检查；
- CLI `--help`；
- metadata 对齐和 hash 检查；
- prompt token 长度检查；
- 参数分组单元测试；
- checkpoint key 契约测试。

### 13.2 Smoke Test

为每个版本生成 8～16 条独立 smoke metadata，运行 2 step 非缓存训练。

检查：

- loss 为有限值；
- forward / backward 正常；
- action tensor 正常进入模型；
- checkpoint 能保存；
- checkpoint 能重新加载；
- 源数据未被修改。

### 13.3 梯度检查

V1 / V2：

- action embedder 有梯度；
- self-attn 有梯度；
- cross-attn 有梯度；
- FFN 有梯度。

V3：

- action embedder 有梯度；
- self-attn 有梯度；
- FFN 有梯度；
- cross-attn LoRA 有梯度；
- cross-attn 原始参数无梯度；
- cross-attn norm_q / norm_k 无梯度；
- `norm3` 无梯度。

LoRA B 常见为零初始化，因此梯度检查至少运行 2 step，确保 A/B 都经过有效更新路径。

### 13.4 固定条件推理

固定：

- input image；
- action sequence；
- seed；
- inference steps。

只改变 Strategy：

- V1：固定 vanilla narration，只替换末尾 Strategy；
- V2：只传不同的 `Strategy(...)`；
- V3：固定 Active / Passive，只替换完整 prompt 中的 Strategy。

对比 checkpoint 重载前后、V3 分离加载与 merged 加载的结果一致性。

## 14. 实施顺序

1. 创建 Rebuttal 目录骨架和 README。
2. 实现 `prepare_metadata.py` 和 metadata 单元测试。
3. 实现统一训练入口、runner 和精确 step 控制。
4. 实现 V1 / V2 full DiT 参数策略。
5. 实现 V3 cross-attn LoRA + 其余 DiT full fine-tune 策略。
6. 实现参数和 optimizer 审计。
7. 实现 V1 / V2 标准 checkpoint。
8. 实现 V3 split checkpoint、恢复和最终合并。
9. 完成三版 2-step 非缓存 smoke test。
10. 实现并验证共享 VAE / 独立 T5 cache。
11. 启动正式训练前输出最终配置、数据 hash、模型指纹、GPU 列表和存储预算。

## 15. 完成标准

只有同时满足以下条件才进入正式训练：

- 所有新增文件都位于 `examples/Rebuttal/`；
- 仓库外部内容没有被修改；
- 两份派生 metadata 可重复生成且 hash 稳定；
- 三版 prompt 与定义逐字一致；
- 三版都实际消费 action 输入；
- V1 / V2 整个 DiT 可训练；
- V3 仅 cross-attn q/k/v/o LoRA 可训练，原始 cross-attn 和 norm3 完全冻结；
- V3 其余 DiT 参数全部可训练；
- 三版统一使用确认后的训练超参数；
- smoke training、checkpoint 重载和固定条件推理全部通过。
