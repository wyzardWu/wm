"""Hardlink-merge 20260731_nativecrop + 20260730_nativecrop into combined_5d.

Rel paths are preserved (clips/chunk_XXX/<session-prefixed-name>.mp4), so the
precomputed cache keys stay valid and the cache shards are linked as-is.
Idempotent: existing links are skipped. Run on the machine that will train.
"""
import os
import sys

import pandas as pd

BASE = "/data/yuzhewu/vrisingroam/processed"
SRCS = ["20260731_nativecrop", "20260730_nativecrop"]
DST = os.path.join(BASE, "combined_5d")


def link_tree(src_root, rel_top):
    src_top = os.path.join(src_root, rel_top)
    if not os.path.isdir(src_top):
        raise SystemExit(f"missing {src_top}")
    n = 0
    for dirpath, _, files in os.walk(src_top):
        rel_dir = os.path.relpath(dirpath, src_root)
        out_dir = os.path.join(DST, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            if f.endswith((".tmp", ".lock")):
                continue
            dst = os.path.join(out_dir, f)
            if not os.path.exists(dst):
                os.link(os.path.join(dirpath, f), dst)
                n += 1
    return n


metas = []
for s in SRCS:
    root = os.path.join(BASE, s)
    for top in ("clips", "actions", "cache/video", "cache/first_frame", "cache/t5"):
        print(f"[merge] {s}/{top}: +{link_tree(root, top)} links", flush=True)
    m = pd.read_csv(os.path.join(root, "metadata.csv"))
    metas.append(m)
    print(f"[merge] {s}: {len(m)} rows", flush=True)

merged = pd.concat(metas, ignore_index=True)
dup = merged.video.duplicated().sum()
if dup:
    raise SystemExit(f"[merge] FATAL: {dup} duplicate clip paths")
merged.to_csv(os.path.join(DST, "metadata.csv"), index=False)

# manifest: configs are identical across sessions; take 0731's.
src_manifest = os.path.join(BASE, SRCS[0], "cache", "manifest.json")
dst_manifest = os.path.join(DST, "cache", "manifest.json")
os.makedirs(os.path.dirname(dst_manifest), exist_ok=True)
if not os.path.exists(dst_manifest):
    os.link(src_manifest, dst_manifest)
print(f"[merge] combined_5d: {len(merged)} rows total", flush=True)

# sanity: every row's video+first_frame+t5 cache entries must exist
import hashlib

def key(rel, ff=False):
    b = f"{rel}|h=480|w=832|nf=101|hdf=16|wdf=16|fps=20"
    if ff:
        b += "|first_frame=1"
    return hashlib.sha256(b.encode()).hexdigest()

missing = 0
for rel in merged.video:
    h, hf = key(rel), key(rel, True)
    if not (os.path.exists(f"{DST}/cache/video/{h[:2]}/{h}.pt")
            and os.path.exists(f"{DST}/cache/first_frame/{hf[:2]}/{hf}.pt")):
        missing += 1
if missing:
    raise SystemExit(f"[merge] FATAL: {missing} rows missing cache entries")
print("[merge] cache completeness check PASSED", flush=True)
