# V Rising 漫游世界模型 (vrisingroam)

用自动采集的 V Rising 邓利农场漫游数据（HF: yuzhewu207/vrisingroam）训练可交互
漫游世界模型。代码基座 ReactiveGWM（Wan2.2-TI2V-5B + 每 block 加性动作注入，
与 Matrix-Game 的 keyboard 注入同构）。

## 布局
- `ReactiveGWM/` — 上游 repo + `DiffSynth-Studio/`（作者未开源的 fork，从
  /nfs/zeqingwang/code/ReactiveGWM/DiffSynth-Studio 复制，见 PROVENANCE.txt）。
  改动：`training/data/profiles.py` 加了 `vrising` profile（5 键，480×832，
  101 帧 @20fps，hold_window=1）。
- `vrising_data/` — 数据管线：timeline(帧对账)/actions(动作栅格化)/filters(坏
  区间)/cut_clips(流式切片)/run_all(下载→切→删编排)/verify(叠印+一致性校验)。
- `scripts/` — precompute_cache.sh（8 卡分 shard）、train_bidirectional.sh。
- `env.sh` — PYTHONPATH + 底模路径（底模在 /nfs/zeqingwang/models/base_model，
  勿复制，直接引用）。
- `data/raw/<session>/` — 日志 + 临时 chunk；`data/processed/<date>/` — clips/
  actions/metadata.csv/cache。

## 关键事实
- 视频 1080p60，日志 视频时间 = HH:MM:SS:FF@60fps，与源视频钟对齐；
  `frame_audit.json` 是分片对账的唯一权威（skip_leading_frames 去重）。
- chunk_000–004 偏移是估算值（start_is_measured=false），默认剔除。
- 0731 session 鼠标日志因盘满丢失 → 动作空间 = W/A/S/D/Mouse0（V Rising 固定
  俯视角，WASD 即完整移动控制）；parquet 里预留 CAM_X/CAM_Y/CAM_ACTIVE 列。
- Python 环境：`/home/yuzhewu/miniconda3/envs/rgwm`（py3.11, torch 2.8.0+cu128，
  从 /nfs/zeqingwang/.conda/envs/qwen_vllm 克隆 + 补装 pyarrow/imageio/modelscope
  等；pytorch 官源/镜像只有 ~300KB/s，勿走 pip 装 torch）。
- 包名兼容：代码 import `ReactiveGWM_Code`，项目根有 `ReactiveGWM_Code -> ReactiveGWM` 软链。
- 本地盘只剩 ~130G：全量处理必须流式（run_all.py），HF 单连接限速 ~2MB/s，
  16 路 range 并行 ~20MB/s。/data 无本用户目录（找管理员建 /data/yuzhewu）。

## 常用命令
```bash
. env.sh
python -m vrising_data.run_all --session 20260731_213546_491 \
  --session_dir data/raw/20260731_213546_491 --out_root data/processed/20260731
python -m vrising_data.verify coherence --session_dir data/raw/20260731_213546_491
bash scripts/precompute_cache.sh data/processed/20260731
bash scripts/train_bidirectional.sh data/processed/20260731 v1_scoped
```
