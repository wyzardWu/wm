"""Offline T5-only precompute for the SF3 `visual` column.

Add-on to `precompute_vae_t5_cache.py`: encodes T5 embeddings for the per-clip
`visual` description (a new CSV column added in `metadata_wo_pure_v5cat_10k_visual.csv`)
and drops them into the existing cache directory's `t5/` shard tree alongside
the existing `prompt`-column embeddings. Hashes are content-derived
(`sha256("t5|v1|<text>")`), so visual and prompt embeddings coexist without
collision.

Outputs:
  - <cache_dir>/t5/<hh>/<hash>.pt  (per unique resolved visual + empty)
  - <cache_dir>/manifest_visual.json  (per-row csv_index -> visual_prompt_hash)

Does NOT touch:
  - <cache_dir>/manifest.json  (existing prompt manifest)
  - <cache_dir>/video/, <cache_dir>/first_frame/  (VAE caches)

Bit-exact T5 encode: imports `encode_prompt_bitexact` from
`precompute_vae_t5_cache.py` so the embedding bytes match what
`WanVideoUnit_PromptEmbedder.encode_prompt` would produce at training time.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_REACTIVE_GWM = Path(__file__).resolve().parents[1]  # examples/ReactiveGWM/
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from data.profiles import SF3 as _PROFILE  # noqa: E402
from data.prompt_utils import resolve_prompt  # noqa: E402

# Reuse helpers from the unified precompute script — single source of truth.
_SCRIPTS_DIR = _REACTIVE_GWM / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from precompute_cache import (  # noqa: E402
    CACHE_MANIFEST_VERSION,
    _shard_path,
    atomic_save,
    encode_prompt_bitexact,
    load_pipeline,
    sha256_file_prefix,
    t5_cache_key,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline T5-only precompute for the SF3 `visual` column"
    )
    p.add_argument("--csv_path", required=True)
    p.add_argument("--cache_dir", required=True,
                   help="Existing cache directory (e.g. clips_5s_cache_v5cat_480x832). "
                        "T5 .pt files are written into <cache_dir>/t5/<hh>/")
    p.add_argument("--prompt_column", default="visual",
                   help="CSV column to encode (default: visual)")
    p.add_argument("--manifest_filename", default="manifest_visual.json",
                   help="Sidecar manifest filename written under cache_dir")
    p.add_argument("--model_paths", required=True,
                   help="JSON list, e.g. '[\"<t5>.pth\",\"<vae>.pth\"]'. "
                        "VAE entry is required by the pipeline loader but its "
                        "compute is unused here. DiT entries are dropped.")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=("bfloat16", "float16", "float32"))
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip prompts whose cached .pt files already exist.")
    p.add_argument("--max_rows", type=int, default=0,
                   help="If > 0, only process the first N CSV rows (smoke test).")
    args = p.parse_args()

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"--cache_dir does not exist: {cache_dir}")
    print(f"[visual_t5] cache_dir={cache_dir}")
    print(f"[visual_t5] prompt_column={args.prompt_column}")

    # Load CSV.
    print(f"[visual_t5] reading {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"[visual_t5] {len(df)} rows, columns={list(df.columns)}")
    if args.prompt_column not in df.columns:
        raise SystemExit(
            f"column {args.prompt_column!r} not found in CSV. "
            f"Available: {list(df.columns)}"
        )
    if args.max_rows > 0 and args.max_rows < len(df):
        df = df.iloc[:args.max_rows].reset_index(drop=True)
        print(f"[visual_t5] --max_rows {args.max_rows}: truncated to {len(df)} rows")
    df = df.assign(_csv_index=df.index.values)

    # Resolve prompt for every row using the shared resolver (mirrors training).
    # Use the SF3 profile so empty/NaN cells fall back to SF3_FIXED_PROMPT.
    resolved = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        resolved.append(resolve_prompt(row, _PROFILE, use_csv_prompt=True,
                                       prompt_column=args.prompt_column))
    unique = sorted(set(resolved) | {""})
    print(f"[visual_t5] unique resolved prompts: {len(unique)} (incl. empty)")

    # Per-row hashes for the manifest.
    rows_meta = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        rel_video = row["video"]
        csv_index = int(row.get("_csv_index", i))
        ph = t5_cache_key(resolved[i])
        rows_meta.append({"csv_index": csv_index, "rel_video": rel_video,
                          "visual_prompt_hash": ph})

    # Load pipeline (T5 + VAE; DiT dropped by load_pipeline).
    pipe = load_pipeline(args.model_paths, args.tokenizer_path,
                         args.device, torch_dtype)

    # Drop VAE — we only need T5 for this script.
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = None
        torch.cuda.empty_cache()

    # T5 encode all unique resolved + empty.
    print(f"[visual_t5] encoding {len(unique)} unique prompts...")
    table = {}
    t_start = time.time()
    n_written = 0
    n_skipped = 0
    for i, prompt in enumerate(unique):
        h = t5_cache_key(prompt)
        out_path = _shard_path(cache_dir, "t5", h)
        table[h] = prompt[:60]
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue
        with torch.no_grad():
            emb = encode_prompt_bitexact(pipe, prompt)
        atomic_save(out_path, emb.to(dtype=pipe.torch_dtype).cpu())
        n_written += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(unique):
            print(f"  T5 {i + 1}/{len(unique)} "
                  f"[{time.time() - t_start:.1f}s, "
                  f"written={n_written}, skipped={n_skipped}]")
    print(f"[visual_t5] done: written={n_written}, skipped={n_skipped}")

    # Sidecar manifest — does not touch cache_dir/manifest.json.
    paths_list = json.loads(args.model_paths)
    t5_path_str = ""
    for entry in paths_list:
        if isinstance(entry, str) and "t5" in entry.lower():
            t5_path_str = entry
            break

    manifest = {
        "version": CACHE_MANIFEST_VERSION,
        "type": "visual_t5_addon",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv_path": str(Path(args.csv_path).resolve()),
        "csv_md5": hashlib.md5(Path(args.csv_path).read_bytes()).hexdigest(),
        "num_rows": int(len(df)),
        "linked_cache_manifest": "manifest.json",
        "config": {
            "cache_dtype": args.torch_dtype,
            "use_csv_prompt": True,
            "prompt_column": args.prompt_column,
        },
        "model_fingerprints": {
            "t5_path": t5_path_str,
            "t5_sha256_prefix": (
                sha256_file_prefix(t5_path_str)
                if t5_path_str and Path(t5_path_str).exists() else ""
            ),
        },
        "rows": rows_meta,
        "prompt_table": table,
        "empty_prompt_hash": t5_cache_key(""),
    }

    manifest_path = cache_dir / args.manifest_filename
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    print(f"[visual_t5] manifest written to {manifest_path}")
    print("[visual_t5] DONE")


if __name__ == "__main__":
    main()
