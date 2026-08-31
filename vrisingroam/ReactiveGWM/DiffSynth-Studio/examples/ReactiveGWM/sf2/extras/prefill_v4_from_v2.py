"""prefill_v4_from_v2.py — 把 v2 cache 中 v4 也需要的 .pt 文件硬链接到 v4 cache 目录。

precompute_cache.py 的 hash 是从 (rel_path, H, W, nf, hdf, wdf) 推导的纯字符串 sha256，
clip 内容不变 → hash 相同。v4 metadata_v4.csv 中的 video / first_frame / prompt 只要在
v2 metadata.csv 出现过，对应 .pt 就在 v2 cache 里。

硬链接（同 inode）零额外磁盘成本；之后跑 precompute --skip_existing 只会补缺失部分。

用法:
    python prefill_v4_from_v2.py \\
        --v2_csv  /.../clips_5s/metadata.csv \\
        --v4_csv  /.../clips_5s/metadata_v4.csv \\
        --v2_cache /.../clips_5s_cache_v2 \\
        --v4_cache /.../clips_5s_cache_v4 \\
        --height 480 --width 608 --num_frames 101
"""
import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def video_cache_key(rel, H, W, nf, hdf, wdf, first_frame=False):
    base = f"{rel}|h={H}|w={W}|nf={nf}|hdf={hdf}|wdf={wdf}|fps=20"
    if first_frame:
        base += "|first_frame=1"
    return sha256_str(base)


def t5_cache_key(prompt):
    return sha256_str(f"t5|v1|{prompt}")


def shard_path(root, kind, h):
    return Path(root) / kind / h[:2] / f"{h}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_csv", required=True)
    ap.add_argument("--v4_csv", required=True)
    ap.add_argument("--v2_cache", required=True)
    ap.add_argument("--v4_cache", required=True)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=608)
    ap.add_argument("--num_frames", type=int, default=101)
    ap.add_argument("--hdf", type=int, default=16)
    ap.add_argument("--wdf", type=int, default=16)
    ap.add_argument("--prompt_column", default="prompt")
    args = ap.parse_args()

    v2_cache = Path(args.v2_cache)
    v4_cache = Path(args.v4_cache)
    v4_cache.mkdir(parents=True, exist_ok=True)

    # 加载 v4 metadata，算所需 hash
    print(f"[load] {args.v4_csv}")
    v4_videos = []
    v4_prompts = set()
    with open(args.v4_csv) as f:
        for r in csv.DictReader(f):
            v4_videos.append(r["video"])
            v4_prompts.add(r.get(args.prompt_column, ""))
    print(f"[load] v4: {len(v4_videos)} videos, {len(v4_prompts)} unique prompts")

    # video / first_frame hash
    needed = {"video": set(), "first_frame": set(), "t5": set()}
    for rel in v4_videos:
        vh = video_cache_key(rel, args.height, args.width, args.num_frames,
                             args.hdf, args.wdf, False)
        ffh = video_cache_key(rel, args.height, args.width, args.num_frames,
                              args.hdf, args.wdf, True)
        needed["video"].add(vh)
        needed["first_frame"].add(ffh)
    for p in v4_prompts:
        needed["t5"].add(t5_cache_key(p))
    for k, s in needed.items():
        print(f"[needed] {k}: {len(s)}")

    # 硬链接 v2 → v4
    n_linked = {"video": 0, "first_frame": 0, "t5": 0}
    n_missing = {"video": 0, "first_frame": 0, "t5": 0}
    for kind, hashes in needed.items():
        for h in hashes:
            src = shard_path(v2_cache, kind, h)
            dst = shard_path(v4_cache, kind, h)
            if dst.exists():
                n_linked[kind] += 1
                continue
            if not src.exists():
                n_missing[kind] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)  # 硬链接，零额外磁盘
                n_linked[kind] += 1
            except OSError as e:
                # 跨 fs 不能硬链接 → fallback 复制
                import shutil
                shutil.copy2(src, dst)
                n_linked[kind] += 1
    print()
    print(f"{'kind':<14s} {'needed':>8s} {'linked':>8s} {'missing':>8s}")
    for kind in needed:
        print(f"{kind:<14s} {len(needed[kind]):>8d} {n_linked[kind]:>8d} {n_missing[kind]:>8d}")

    print(f"\n[done] v4 cache prefill from v2 → {v4_cache}")
    print(f"  next: run precompute_cache.py with --cache_root={v4_cache} --skip_existing")


if __name__ == "__main__":
    main()
