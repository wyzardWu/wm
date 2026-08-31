"""Stream-process a whole session within a small local-disk budget.

For each requested chunk: download (aria2c multi-connection if available,
else curl) -> cut clips -> delete the raw chunk. Raw chunks are ~7 GB each and
processed output is ~2% of that, so peak disk usage stays near
`parallel_downloads * 7GB` + accumulated clips.

Usage:
  python -m vrising_data.run_all \
      --session 20260731_213546_491 \
      --session_dir data/raw/20260731_213546_491 \
      --out_root data/processed/20260731 \
      --start 5 --end 66
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HF_BASE = "https://huggingface.co/datasets/yuzhewu207/vrisingroam/resolve/main"


def _fetch_part(url: str, part: str, lo: int, hi: int, tries: int = 6) -> None:
    """One byte-range with resume: appends from wherever the part left off.
    `-f` keeps HTTP error bodies (429/5xx pages) out of the part file."""
    want = hi - lo + 1
    for attempt in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > want:  # stale part from a different layout — start over
            os.remove(part)
            have = 0
        if have == want:
            return
        with open(part, "ab") as f:
            subprocess.run(
                ["curl", "-sfL", "--retry", "3", "-r", f"{lo + have}-{hi}", url],
                stdout=f)
        time.sleep(2 * attempt)
    if (os.path.getsize(part) if os.path.exists(part) else 0) != want:
        raise RuntimeError(f"part incomplete after {tries} tries: {part}")


def download(url: str, dest: str, n_conn: int = 32) -> None:
    """Ranged parallel download — HF rate-limits per connection (~2 MB/s) and
    the cap is per-flow (measured 08/04: 32 conns ≈ 2x throughput of 16), so
    32 ranges give ~40 MB/s. Each range resumes independently across retries
    and process restarts."""
    tmp = dest + ".part"
    size = int(subprocess.run(
        ["curl", "-sIL", url], capture_output=True, text=True, check=True,
    ).stdout.lower().rsplit("content-length:", 1)[1].split()[0])
    step = -(-size // n_conn)
    spans = [(i * step, min((i + 1) * step, size) - 1) for i in range(n_conn)]
    spans = [(lo, hi) for lo, hi in spans if lo <= hi]
    with ThreadPoolExecutor(len(spans)) as ex:
        futs = [ex.submit(_fetch_part, url, f"{tmp}.{i:02d}", lo, hi)
                for i, (lo, hi) in enumerate(spans)]
        for fut in futs:
            fut.result()
    with open(tmp, "wb") as out:
        for i in range(len(spans)):
            part = f"{tmp}.{i:02d}"
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 22)
            os.remove(part)
    if os.path.getsize(tmp) != size:
        raise RuntimeError(f"size mismatch after merge: {dest}")
    os.replace(tmp, dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--session_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--start", type=int, default=5)
    p.add_argument("--end", type=int, default=66)
    # Keep raw 1080p chunks after cutting (~450 GB total). data/ lives on the
    # 28 TB /data array, so retaining them makes any future re-cut (crop/fps
    # changes) a 2 h job instead of a full re-download.
    p.add_argument("--keep_raw", action="store_true")
    p.add_argument("--stride", type=int, default=101)
    args = p.parse_args()

    video_dir = os.path.join(args.session_dir, "video")
    os.makedirs(video_dir, exist_ok=True)
    done_marker_dir = os.path.join(args.out_root, ".done")
    os.makedirs(done_marker_dir, exist_ok=True)

    def chunk_name(i):
        return f"chunk_{i:03d}.mp4"

    def is_done(i):
        return os.path.exists(os.path.join(done_marker_dir, chunk_name(i) + ".done"))

    # Single-worker pool: while chunk i is being cut, the worker downloads
    # chunk i+1, so the ~5 min download and ~2 min cut overlap.
    dl_pool = ThreadPoolExecutor(1)

    def start_download(i):
        path = os.path.join(video_dir, chunk_name(i))
        if os.path.exists(path):
            return None
        return dl_pool.submit(
            download, f"{HF_BASE}/{args.session}/video/{chunk_name(i)}", path)

    pending = list(range(args.start, args.end + 1))
    for round_no in range(1, 6):  # up to 5 passes over failed chunks
        failed = []
        prefetched: dict[int, object] = {}
        for pos, i in enumerate(pending):
            name = chunk_name(i)
            marker = os.path.join(done_marker_dir, name + ".done")
            if os.path.exists(marker):
                print(f"[run_all] {name}: already processed, skip", flush=True)
                continue
            path = os.path.join(video_dir, name)
            try:
                fut = prefetched.pop(i, None)
                if fut is None and not os.path.exists(path):
                    print(f"[run_all] downloading {name}", flush=True)
                    fut = start_download(i)
                # queue the next unfinished chunk behind this one
                nxt = next((j for j in pending[pos + 1:]
                            if not is_done(j) and j not in prefetched), None)
                if nxt is not None:
                    f2 = start_download(nxt)
                    if f2 is not None:
                        prefetched[nxt] = f2
                if fut is not None:
                    fut.result()
                print(f"[run_all] cutting {name}", flush=True)
                subprocess.run(
                    [sys.executable, "-m", "vrising_data.cut_clips",
                     "--session_dir", args.session_dir, "--out_root", args.out_root,
                     "--chunks", name, "--stride", str(args.stride)],
                    check=True,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                open(marker, "w").close()
                if not args.keep_raw:
                    os.remove(path)
                    print(f"[run_all] removed raw {name}", flush=True)
            except Exception as e:  # noqa: BLE001 — keep going, retry later
                print(f"[run_all] {name} FAILED (round {round_no}): {e}", flush=True)
                failed.append(i)
        if not failed:
            break
        print(f"[run_all] round {round_no}: {len(failed)} chunks failed, "
              f"retrying in 60s", flush=True)
        pending = failed
        time.sleep(60)
    if pending and failed:
        raise SystemExit(f"[run_all] gave up on chunks: {failed}")


if __name__ == "__main__":
    main()
