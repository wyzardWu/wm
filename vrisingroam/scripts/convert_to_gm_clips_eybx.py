"""Convert the EYBX cached dataset -> zhiyang cf_distill_3stage precomputed format.

Same contract as convert_to_gm_clips.py (V Rising), with the EYBX combined
context: per latent frame the CA context is [action 16 tok] ⊕ [scene 16 tok].
We express each (action_id, scene_id) pair as ONE combined "sentence" key
("<action sentence> || <scene sentence>") whose text_table embedding is the
concatenation cat(action_row, scene_row) -> [32, 4096]. zhiyang's
CrossAttention accepts arbitrary L, so his code is unchanged.

Binning replicates train.py exactly: adaptive_max_pool1d over the 6-channel
keyboard tensor; ids from cols 0-4 (bitmask), scene ids from col 5
(0..2 = worded scenes, 3 = null "the scene is somewhere").

Usage:
  . env.sh && python scripts/convert_to_gm_clips_eybx.py \
      --data_root /data/yuzhewu/eybxroam/overfit4 \
      --out /data/yuzhewu/vrisingroam/distill/data/eybx_F26_v1 \
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
from ReactiveGWM_Code.training.bidirectional.cached_dataset import CachedReactiveGWMDataset

NF = 101
F_LAT = (NF - 1) // 4 + 1  # 26
TABLES = "/data/yuzhewu/eybxroam/tables"
ACTION_NEUTRAL_ID = 32


def bin_ids(ka: torch.Tensor):
    """train.py table-mode binning: action bitmask ids + scene ids (null=3 legal)."""
    if ka.dim() == 2:
        ka = ka.unsqueeze(0)
    binned = torch.nn.functional.adaptive_max_pool1d(
        ka.transpose(1, 2).float(), output_size=F_LAT,
    ).transpose(1, 2)
    b = (binned[..., :5] > 0.5).long()
    ids = (b[..., 0] * 1 + b[..., 1] * 2 + b[..., 2] * 4
           + b[..., 3] * 8 + b[..., 4] * 16)[0]
    sids = binned[0, :, 5].round().long().clamp(0, 3)
    return ids, sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--num_workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    at = torch.load(f"{TABLES}/eybx_action_table.pt", map_location="cpu")
    st = torch.load(f"{TABLES}/eybx_scene_table.pt", map_location="cpu")
    act_sent, act_tbl = at["sentences"], at["table"]
    scn_sent, scn_tbl = st["sentences"], st["table"]

    def key(a, s):
        return f"{act_sent[a]} || {scn_sent[s]}"

    emb, text_len = {}, {}
    for a in range(len(act_sent)):
        for s in range(len(scn_sent)):
            k = key(a, s)
            if k in emb:
                continue          # null action sentence repeats across rows
            emb[k] = torch.cat([act_tbl[a], scn_tbl[s]], dim=0).to(torch.bfloat16)
            text_len[k] = act_tbl.shape[1] + scn_tbl.shape[1]      # 32
    torch.save({"emb": emb, "text_len": text_len},
               os.path.join(args.out, "text_table.pt"))
    print(f"text_table: {len(emb)} combined keys, L={next(iter(text_len.values()))}")

    profile = get_profile("eybx")
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
    todo = [i for i in range(args.start, end)
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
        if lat.dim() == 5: lat = lat[0]
        if ff.dim() == 5: ff = ff[0]
        assert lat.shape[1] == F_LAT, f"latent F {lat.shape} != {F_LAT}"
        lat = lat.clone()
        lat[:, 0:1] = ff[:, 0:1]                      # i2v fuse (wan_video.py:336)
        ids, sids = bin_ids(data["action"])
        prompts = [key(int(a), int(s)) for a, s in zip(ids, sids)]
        # dropout track: null action, scene kept (stage1 runs dropout=0; format compat)
        prompts_drop = [key(ACTION_NEUTRAL_ID, int(s)) for s in sids]
        torch.save({
            "latent": lat.to(torch.bfloat16).contiguous(),
            "prompts": prompts,
            "prompts_bossdrop": prompts_drop,
            "fight": "eybx",
            "start_cell": i,
            "boss": None,
        }, os.path.join(args.out, f"clip_{i:06d}.pt"))
        man.write(json.dumps({"i": i}) + "\n")
        if k % 200 == 0:
            man.flush()
            print(f"[{k}/{len(todo)}] clip_{i:06d}", flush=True)
    man.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
