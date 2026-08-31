"""Convert our ReactiveGWM cached dataset -> zhiyang cf_distill_3stage precomputed format.

Emits clip_%06d.pt ({latent [48,F,h,w] bf16, prompts[F], prompts_bossdrop[F], fight,
start_cell, boss}) + text_table.pt ({emb: {sentence: [L,4096] bf16}, text_len}), the exact
shapes gamemaster/data/precomputed.py::PrecomputedRenderDataset consumes.

Fidelity rules (teacher-student conditioning consistency):
  - latent = cached video latents with frame0 REPLACED by the cached first-frame latent
    (reproduces wan_video.py:336 `latents[:,:,0:1] = first_frame_latents`, i.e. exactly
    what the bidirectional teacher saw as GT).
  - prompts = v3 action sentences via the SAME binning as train.py table mode
    (adaptive_max_pool1d over the profile keyboard tensor -> 5-bit combo id -> table row).
  - prompts_bossdrop = neutral sentence everywhere (stage1 runs boss_dropout=0; kept for
    format compatibility).

Usage (node2):
  . env.sh && python scripts/convert_to_gm_clips.py \
      --data_root /data/yuzhewu/vrisingroam/processed/combined_5d \
      --metadata metadata_train.csv \
      --table /data/yuzhewu/vrisingroam/processed/action_context_table_v3.pt \
      --out /data/yuzhewu/vrisingroam/distill/data/vrising_F26_v3 \
      [--start 0 --end -1] [--num_workers 16]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ReactiveGWM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ReactiveGWM" / "DiffSynth-Studio"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ReactiveGWM_Code.training.data.profiles import get_profile
from ReactiveGWM_Code.training.data.action_text import ALL_SENTENCES, NEUTRAL_ID
from ReactiveGWM_Code.training.bidirectional.cached_dataset import CachedReactiveGWMDataset

NF = 101
F_LAT = (NF - 1) // 4 + 1  # 26


def combo_ids(ka: torch.Tensor) -> torch.Tensor:
    """EXACT replica of train.py table-mode binning (v3: first 5 columns only)."""
    if ka.dim() == 2:
        ka = ka.unsqueeze(0)
    binned = torch.nn.functional.adaptive_max_pool1d(
        ka.transpose(1, 2).float(), output_size=F_LAT,
    ).transpose(1, 2)
    b = (binned[..., :5] > 0.5).long()
    ids = (b[..., 0] * 1 + b[..., 1] * 2 + b[..., 2] * 4
           + b[..., 3] * 8 + b[..., 4] * 16)
    return ids[0]  # [F_LAT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--metadata", default="metadata_train.csv")
    ap.add_argument("--table", required=True, help="action_context_table_v3.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--num_workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # text_table.pt (idempotent, tiny)
    tbl = torch.load(args.table, map_location="cpu")
    assert tbl["sentences"] == ALL_SENTENCES, "table/action_text sentence mismatch"
    emb = {s: tbl["table"][i].to(torch.bfloat16) for i, s in enumerate(ALL_SENTENCES)}
    text_len = {s: int(tbl["table_len"]) for s in ALL_SENTENCES}
    torch.save({"emb": emb, "text_len": text_len}, os.path.join(args.out, "text_table.pt"))
    neutral = ALL_SENTENCES[NEUTRAL_ID]

    profile = get_profile("vrising")
    ds = CachedReactiveGWMDataset(
        profile=profile,
        base_path=args.data_root,
        metadata_path=os.path.join(args.data_root, args.metadata),
        cache_root=os.path.join(args.data_root, "cache"),
        num_frames=NF, height=480, width=832,
        action_hold_window=1,
        use_csv_prompt=False,
        strict=False,
    )
    n = len(ds)
    end = n if args.end < 0 else min(args.end, n)
    idxs = list(range(args.start, end))
    # skip already-converted (resume support)
    todo = [i for i in idxs
            if not os.path.exists(os.path.join(args.out, f"clip_{i:06d}.pt"))]
    print(f"dataset {n} items; range [{args.start},{end}); todo {len(todo)}", flush=True)

    sub = torch.utils.data.Subset(ds, todo)
    dl = DataLoader(sub, batch_size=None, num_workers=args.num_workers,
                    collate_fn=lambda x: x)

    man = open(os.path.join(args.out, f"convert_manifest_{args.start}.jsonl"), "a")
    for k, data in enumerate(dl):
        i = todo[k]
        lat = data["__cached_input_latents"]
        ff = data["__cached_first_frame_latents"]
        if lat.dim() == 5:
            lat = lat[0]
        if ff.dim() == 5:
            ff = ff[0]
        assert lat.shape[1] == F_LAT, f"latent F {lat.shape} != {F_LAT}"
        lat = lat.clone()
        lat[:, 0:1] = ff[:, 0:1]                      # i2v fuse (wan_video.py:336)
        ids = combo_ids(data["action"])
        prompts = [ALL_SENTENCES[int(t)] for t in ids]
        rel = f'{data.get("session","?")}/{data.get("chunk","?")}'
        torch.save({
            "latent": lat.to(torch.bfloat16).contiguous(),
            "prompts": prompts,
            "prompts_bossdrop": [neutral] * F_LAT,
            "fight": "vrising",
            "start_cell": i,
            "boss": None,
        }, os.path.join(args.out, f"clip_{i:06d}.pt"))
        man.write(json.dumps({"i": i, "rel": str(rel)[:200]}) + "\n")
        if k % 500 == 0:
            man.flush()
            print(f"[{k}/{len(todo)}] clip_{i:06d}", flush=True)
    man.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
