"""Rebuild SF3 cache with use_csv_prompt=False (fixed-prompt training).

Existing v5cat cache (built via run_cache_8shard_v5cat.sh) has T5 entries for
~7300 unique per-row prompts and a manifest.config with use_csv_prompt=True.
For SF_wo_cross training (cross_attn frozen, fixed prompt) we want a cache
where every row's prompt_hash points to T5(SF3_FIXED_PROMPT).

Plan:
  1. Hardlink video/ + first_frame/ entire trees from source cache_dir into
     dest cache_dir (zero disk increment — same inode).
  2. Encode T5 for SF3_FIXED_PROMPT + empty string. Save to dest/t5/.
  3. Write a new manifest.json:
     - config.use_csv_prompt = False
     - config.prompt_column = "prompt"  (unused but kept for schema compat)
     - rows: same csv_index/rel_video/video_hash/first_frame_hash as source,
             but every row's prompt_hash = t5_cache_key(SF3_FIXED_PROMPT)
     - empty_prompt_hash = t5_cache_key("")
     - failed_rows: copied verbatim from source manifest

Idempotent: --skip_existing makes hardlink + T5 encode safe to re-run.

Typical wall time: <1 minute (hardlinks are O(file count), 2 T5 encodes,
no per-row VAE work).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_REACTIVE_GWM = Path(__file__).resolve().parents[1]  # examples/ReactiveGWM/
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from diffsynth.pipelines.reactive_gwm import ModelConfig, ReactiveGWMPipeline  # noqa: E402

from data.profiles import SF3 as _PROFILE  # noqa: E402
SF3_FIXED_PROMPT = _PROFILE.fixed_prompt


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def t5_cache_key(resolved_prompt: str) -> str:
    return sha256_str(f"t5|v1|{resolved_prompt}")


def _shard_path(root: Path, kind: str, h: str) -> Path:
    return root / kind / h[:2] / f"{h}.pt"


def encode_prompt_bitexact(pipe, prompt: str) -> torch.Tensor:
    """Copy of WanVideoUnit_PromptEmbedder.encode_prompt — same as
    examples/sf3/scripts/precompute_vae_t5_cache.py."""
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


def hardlink_tree(src: Path, dst: Path, skip_existing: bool) -> int:
    """Recursively hardlink every regular file under src/ to dst/. Returns count."""
    n = 0
    for src_path in src.rglob("*"):
        if not src_path.is_file():
            continue
        rel = src_path.relative_to(src)
        dst_path = dst / rel
        if skip_existing and dst_path.exists():
            n += 1
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src_path, dst_path)
        except FileExistsError:
            pass  # raced with another helper
        except OSError as e:
            # Fall back to copy if hardlink not supported (cross-device, etc.)
            print(f"  [warn] hardlink {src_path.name} failed ({e}); copying")
            import shutil
            shutil.copy2(src_path, dst_path)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(
        description="Rebuild SF3 cache with fixed-prompt T5 (hardlinks video/ff)"
    )
    p.add_argument("--src_cache_dir", required=True,
                   help="Existing cache (built with use_csv_prompt=True). "
                        "Must contain manifest.json + video/ + first_frame/ + t5/.")
    p.add_argument("--dst_cache_dir", required=True,
                   help="New cache to write. Will be created if missing.")
    p.add_argument("--model_paths", required=True,
                   help="JSON list, e.g. '[t5_path, vae_path]' — VAE entries are "
                        "ignored; we only use T5 + tokenizer here.")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=("bfloat16", "float16", "float32"))
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip files that already exist in dst (safe re-run).")
    p.add_argument("--no_hardlink", action="store_true",
                   help="Skip the video/first_frame hardlink phase. Useful if "
                        "you've already linked them once and just want to redo T5.")
    args = p.parse_args()

    src = Path(args.src_cache_dir)
    dst = Path(args.dst_cache_dir)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[rebuild] src={src}")
    print(f"[rebuild] dst={dst}")

    src_manifest_path = src / "manifest.json"
    if not src_manifest_path.exists():
        raise SystemExit(f"src manifest missing: {src_manifest_path}")
    src_manifest = json.loads(src_manifest_path.read_text())
    print(f"[rebuild] src manifest: {len(src_manifest.get('rows', []))} rows")

    # 1. Hardlink video/ + first_frame/.
    if not args.no_hardlink:
        for kind in ("video", "first_frame"):
            t0 = time.time()
            sub = src / kind
            if not sub.exists():
                print(f"[rebuild] [warn] {sub} missing — skipping {kind}")
                continue
            n = hardlink_tree(sub, dst / kind, args.skip_existing)
            print(f"[rebuild] {kind}: linked/exist {n} files in {time.time() - t0:.1f}s")
    else:
        print("[rebuild] --no_hardlink: skipping video/first_frame phase")

    # 2. Encode T5 for fixed prompt + empty.
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[args.torch_dtype]
    paths = json.loads(args.model_paths)
    model_configs = [ModelConfig(path=p) for p in paths]
    tokenizer_config = ModelConfig(path=args.tokenizer_path)
    print("[rebuild] loading T5 + tokenizer (VAE/DiT loaded then dropped)...")
    pipe = ReactiveGWMPipeline.from_pretrained(
        torch_dtype=torch_dtype, device=args.device,
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )
    if hasattr(pipe, "dit") and pipe.dit is not None:
        pipe.dit = None
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = None
    torch.cuda.empty_cache()

    fixed_hash = t5_cache_key(SF3_FIXED_PROMPT)
    empty_hash = t5_cache_key("")
    print(f"[rebuild] SF3_FIXED_PROMPT={SF3_FIXED_PROMPT!r} -> hash {fixed_hash[:16]}...")
    print(f"[rebuild] empty hash {empty_hash[:16]}...")

    for hsh, prompt in [(fixed_hash, SF3_FIXED_PROMPT), (empty_hash, "")]:
        out_path = _shard_path(dst, "t5", hsh)
        if args.skip_existing and out_path.exists():
            print(f"[rebuild] T5 {hsh[:16]} already cached, skipping")
            continue
        with torch.no_grad():
            emb = encode_prompt_bitexact(pipe, prompt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
        torch.save(emb.to(dtype=torch_dtype).cpu(), tmp)
        os.replace(tmp, out_path)
        print(f"[rebuild] wrote T5 {hsh[:16]} <- {prompt!r} ({emb.shape})")

    # 3. Write new manifest. Reuse rows + failed_rows + model fingerprints from
    # source, but rewrite prompt_hash + config.use_csv_prompt.
    new_rows = []
    for r in src_manifest.get("rows", []):
        new_rows.append({
            "csv_index": r["csv_index"],
            "rel_video": r["rel_video"],
            "video_hash": r["video_hash"],
            "first_frame_hash": r["first_frame_hash"],
            "prompt_hash": fixed_hash,
        })

    src_cfg = src_manifest.get("config", {})
    new_cfg = dict(src_cfg)
    new_cfg["use_csv_prompt"] = False
    # prompt_column is irrelevant when use_csv_prompt=False but the manifest
    # check still compares it byte-exact. Force it to "prompt" to match what
    # train.py defaults to so the launch shell doesn't need to pass --prompt_column.
    new_cfg["prompt_column"] = "prompt"
    new_cfg["fixed_prompt"] = SF3_FIXED_PROMPT

    new_manifest = {
        "version": src_manifest.get("version", 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv_path": src_manifest.get("csv_path"),
        "csv_md5": src_manifest.get("csv_md5"),
        "num_rows": src_manifest.get("num_rows"),
        "dataset_base": src_manifest.get("dataset_base"),
        "config": new_cfg,
        "model_fingerprints": src_manifest.get("model_fingerprints", {}),
        "rows": new_rows,
        "prompt_table": {
            fixed_hash: SF3_FIXED_PROMPT[:60],
            empty_hash: "",
        },
        "empty_prompt_hash": empty_hash,
        "failed_rows": src_manifest.get("failed_rows", []),
        "rebuilt_from": str(src.resolve()),
    }
    out_manifest = dst / "manifest.json"
    tmp = out_manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_manifest, indent=2, ensure_ascii=False))
    os.replace(tmp, out_manifest)
    print(f"[rebuild] manifest -> {out_manifest}")
    print(f"[rebuild] DONE. dst={dst}")


if __name__ == "__main__":
    main()
