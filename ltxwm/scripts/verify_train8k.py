"""抽样校验 train8k_precomp:配对完整性 + latent 统计 + conditions 结构 + action_ids 一致性."""
import json, os, random, sys
import numpy as np
import torch

LTX = "/data/yuzhewu/ltxwm"
ROOT = f"{LTX}/data/train8k_precomp"
MANIFEST = f"{LTX}/data/train8k_minus4.jsonl"
TEST = f"{LTX}/data/test_heldout4.jsonl"
ACTIONS = f"{LTX}/data/actions"
TRAIN2K = f"{LTX}/data/train2000.jsonl"

rows = [json.loads(l) for l in open(MANIFEST)]
test_rows = [json.loads(l) for l in open(TEST)]
train2k = set(json.loads(l)["video"] for l in open(TRAIN2K))
rel = lambda v: v.lstrip("/").replace(".mp4", ".pt")

# 1) 全量配对:manifest 每条都有 conditions+latents;目录里无多余文件;测试 clip 不在内
missing = [r["video"] for r in rows
           if not (os.path.exists(f"{ROOT}/conditions/{rel(r['video'])}")
                   and os.path.exists(f"{ROOT}/latents/{rel(r['video'])}"))]
n_cond = sum(len(fs) for _, _, fs in os.walk(f"{ROOT}/conditions"))
n_lat = sum(len(fs) for _, _, fs in os.walk(f"{ROOT}/latents"))
leak = [r["video"] for r in test_rows if os.path.exists(f"{ROOT}/latents/{rel(r['video'])}")]
print(f"[pair] manifest={len(rows)} cond_files={n_cond} lat_files={n_lat} missing={len(missing)} test_leak={len(leak)}")
assert not missing and not leak and n_cond == n_lat == len(rows) == 7996, "PAIRING FAIL"

# 2) 抽样:8 条新算 + 4 条硬链接老件
random.seed(0)
new = [r for r in rows if r["video"] not in train2k]
old = [r for r in rows if r["video"] in train2k]
sample = [("new", r) for r in random.sample(new, 8)] + [("old", r) for r in random.sample(old, 4)]

sys.path.insert(0, f"{LTX}/scripts")
from bin_abot_actions import bin_one  # 重跑分箱逻辑做端到端对账

stats, n_move_active, n_view_active = {"new": [], "old": []}, 0, 0
for tag, r in sample:
    v = r["video"]
    lat = torch.load(f"{ROOT}/latents/{rel(v)}", map_location="cpu", weights_only=True)
    t = lat["latents"] if isinstance(lat, dict) and "latents" in lat else lat
    if isinstance(t, dict):
        t = next(x for x in t.values() if torch.is_tensor(x) and x.ndim >= 4)
    lstd, lmean = t.float().std().item(), t.float().mean().item()
    nan = torch.isnan(t.float()).any().item()
    stats[tag].append(lstd)

    cond = torch.load(f"{ROOT}/conditions/{rel(v)}", map_location="cpu", weights_only=True)
    cap = cond["video_prompt_embeds"]; msk = cond["prompt_attention_mask"]; ids = cond["action_ids"]
    ok_shape = tuple(cap.shape) == (1024, 4096) and tuple(ids.shape) == (16, 2)
    in_range = bool((ids >= 0).all() and (ids <= 8).all())

    npy_rel = os.path.relpath(v, "/nfs/danze/data/abot/clips").replace(".mp4", ".ltxids.npy")
    disk_ids = np.load(f"{ACTIONS}/{npy_rel}")
    match_disk = np.array_equal(ids.numpy(), disk_ids)
    rebin = bin_one(v.replace(".mp4", ".npy"))
    match_rebin = np.array_equal(ids.numpy(), np.asarray(rebin, dtype=np.int8)) if rebin is not None else None

    mv, vw = ids[:, 0].numpy(), ids[:, 1].numpy()
    n_move_active += int((mv > 0).sum()); n_view_active += int((vw > 0).sum())
    print(f"[{tag}] {os.path.basename(v)[:20]} lat{tuple(t.shape)} std={lstd:.3f} mean={lmean:+.3f} "
          f"nan={nan} cond_ok={ok_shape} ids_range={in_range} ids==npy={match_disk} ids==rebin={match_rebin} "
          f"move={mv.tolist()} view={vw.tolist()} mask_sum={int(msk.sum())}")
    assert ok_shape and in_range and match_disk and not nan, f"SAMPLE FAIL {v}"

print(f"[stats] latent std new={np.mean(stats['new']):.3f}±{np.std(stats['new']):.3f} "
      f"old={np.mean(stats['old']):.3f}±{np.std(stats['old']):.3f}")
print(f"[actions] 12 clip × 16 帧: move 非静止 {n_move_active}/192, view 非静止 {n_view_active}/192")
print("VERIFY_OK")
