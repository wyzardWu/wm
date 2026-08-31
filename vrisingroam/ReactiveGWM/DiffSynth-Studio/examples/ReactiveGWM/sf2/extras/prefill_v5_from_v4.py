"""prefill_v5_from_v4.py — 把 v4 cache 中 v5 也需要的 .pt 文件 COPY 到 v5 cache 目录。

precompute_cache.py 的 hash 是从 (rel_path, H, W, nf, hdf, wdf) 推导，video 内容不变 → hash 相同。
v5 metadata_v5.csv 中的 video / first_frame，只要在 v4 metadata_v4.csv 出现过，对应 .pt 就在 v4 cache 里。

注意: 与 prefill_v4_from_v2.py 不同，这里用 shutil.copy2 (复制) 而不是 os.link (硬链接)，
v4 与 v5 cache 完全独立，避免 cross-contamination。

t5 prompt 不复用 (v5 prompt 格式是 3-cat paraphrase，与 v4 sub-strategy 不同)。

用法:
    python prefill_v5_from_v4.py \\
        --v4_csv  /.../clips_5s/metadata_v4.csv \\
        --v5_csv  /.../clips_5s/metadata_v5.csv \\
        --v4_cache /.../clips_5s_cache_v4 \\
        --v5_cache /.../clips_5s_cache_v5 \\
        --height 480 --width 608 --num_frames 101
"""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def video_cache_key(rel, H, W, nf, hdf, wdf, first_frame=False):
    base = f"{rel}|h={H}|w={W}|nf={nf}|hdf={hdf}|wdf={wdf}|fps=20"
    if first_frame:
        base += "|first_frame=1"
    return sha256_str(base)


def shard_path(root, kind, h):
    return Path(root) / kind / h[:2] / f"{h}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4_csv", required=True)
    ap.add_argument("--v5_csv", required=True)
    ap.add_argument("--v4_cache", required=True)
    ap.add_argument("--v5_cache", required=True)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=608)
    ap.add_argument("--num_frames", type=int, default=101)
    ap.add_argument("--hdf", type=int, default=16)
    ap.add_argument("--wdf", type=int, default=16)
    args = ap.parse_args()

    v4_cache = Path(args.v4_cache)
    v5_cache = Path(args.v5_cache)
    v5_cache.mkdir(parents=True, exist_ok=True)

    # v4 中已有的 video set (路径 → 用于判断是否在 v4 cache 里)
    v4_videos = set()
    with open(args.v4_csv) as f:
        for r in csv.DictReader(f):
            v4_videos.add(r["video"])
    print(f"[load] v4 has {len(v4_videos)} videos")

    # v5 需要的 hash (只看 v4 也有的 video — 这些可以复制)
    print(f"[load] {args.v5_csv}")
    needed_video = []
    needed_ff = []
    n_v5 = 0
    n_v5_in_v4 = 0
    with open(args.v5_csv) as f:
        for r in csv.DictReader(f):
            n_v5 += 1
            rel = r["video"]
            if rel not in v4_videos:
                continue
            n_v5_in_v4 += 1
            vh = video_cache_key(rel, args.height, args.width, args.num_frames,
                                 args.hdf, args.wdf, False)
            ffh = video_cache_key(rel, args.height, args.width, args.num_frames,
                                  args.hdf, args.wdf, True)
            needed_video.append(vh)
            needed_ff.append(ffh)
    print(f"[load] v5: {n_v5} rows total, {n_v5_in_v4} also in v4 (copy candidates)")

    # 复制 (跳过已存在)
    n_copied = {"video": 0, "first_frame": 0}
    n_existing = {"video": 0, "first_frame": 0}
    n_missing_in_v4 = {"video": 0, "first_frame": 0}

    for kind, hashes in [("video", needed_video), ("first_frame", needed_ff)]:
        for h in hashes:
            src = shard_path(v4_cache, kind, h)
            dst = shard_path(v5_cache, kind, h)
            if dst.exists():
                n_existing[kind] += 1
                continue
            if not src.exists():
                n_missing_in_v4[kind] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_copied[kind] += 1

    print()
    print(f"{'kind':<14s} {'needed':>8s} {'copied':>8s} {'existing':>8s} {'missing_in_v4':>14s}")
    for kind in ("video", "first_frame"):
        total = n_copied[kind] + n_existing[kind] + n_missing_in_v4[kind]
        print(f"{kind:<14s} {total:>8d} {n_copied[kind]:>8d} {n_existing[kind]:>8d} {n_missing_in_v4[kind]:>14d}")

    print(f"\n[done] v5 cache prefilled from v4 → {v5_cache}")
    print(f"  v5 rows that need fresh video/first_frame compute: {n_v5 - n_v5_in_v4}")
    print(f"  next: run precompute_cache.py --cache_root={v5_cache} --skip_existing")


if __name__ == "__main__":
    main()
