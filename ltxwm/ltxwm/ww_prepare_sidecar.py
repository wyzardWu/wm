"""WildWorld sidecar:往 precompute 的 conditions .pt 里注入 ww_action_ids [16,3]。

每潜帧 k 对应源帧窗 [k*8, k*8+8)(clip 取前 121 帧,潜帧 16):
  player id: 窗内玩家三元组 (weapon,bank,motion) 众数 -> player_keys 索引(缺->0)
  boss   id: 窗内怪物三元组 (type,bank,motion) 众数  -> boss_keys 索引(无怪/缺->0)
  cam    id: 窗内相机 yaw/pitch 变化量离散到 9 类(idle,J,L,I,K,JI,JK,LI,LK)
幂等:已有 ww_action_ids 的文件跳过。

Usage: python ww_prepare_sidecar.py --precomp /data/yuzhewu/ltxwm/data/ww_pilot_precomp \
           [--yaw_thresh 1.2 --pitch_thresh 0.8] [--workers 16]
"""
import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch

TABLES = "/data/yuzhewu/ltxwm/tables/ww_action_tables_v1.pt"
PILOT = "/data/yuzhewu/ltxwm/data/ww_pilot"
NF, STRIDE, NLAT = 121, 8, 16
_G = {}


def quat_yaw_pitch(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # OpenGL 相机看向 -Z:forward = R @ [0,0,-1]
    fx = -(2 * (x * z + y * w))
    fy = -(2 * (y * z - x * w))
    fz = -(1 - 2 * (x * x + y * y))
    yaw = np.degrees(np.arctan2(fx, -fz))
    pitch = np.degrees(np.arcsin(np.clip(fy, -1, 1)))
    return yaw, pitch


def cam_id(dyaw, dpitch, yt, pt):
    # v3.1 view 表顺序: ["", J(左), L(右), I(上), K(下), JI, JK, LI, LK]
    h = "J" if dyaw > yt else ("L" if dyaw < -yt else "")
    v = "I" if dpitch > pt else ("K" if dpitch < -pt else "")
    return {"": 0, "J": 1, "L": 2, "I": 3, "K": 4, "JI": 5, "JK": 6,
            "LI": 7, "LK": 8}[h + v]


def process_one(args_tuple):
    cond_path, state_path, yt, pt = args_tuple
    try:
        d = torch.load(cond_path, map_location="cpu", weights_only=True)
        if "ww_action_ids" in d:
            return 0
        z = np.load(state_path)
        pkey2i, bkey2i = _G["pkey2i"], _G["bkey2i"]
        yaw, pitch = quat_yaw_pitch(z["cam_rot"][:NF])
        yaw = np.unwrap(np.radians(yaw)) * 180 / np.pi
        ids = np.zeros((NLAT, 3), dtype=np.int32)
        for k in range(NLAT):
            s0, s1 = k * STRIDE, min(k * STRIDE + STRIDE, NF)
            trip = Counter(zip(z["weapon"][s0:s1], z["p_bank"][s0:s1],
                               z["p_motion"][s0:s1])).most_common(1)[0][0]
            ids[k, 0] = pkey2i.get(str(tuple(int(v) for v in trip)), 0)
            btr = Counter(zip(z["m_type"][s0:s1], z["m_bank"][s0:s1],
                              z["m_motion"][s0:s1])).most_common(1)[0][0]
            ids[k, 1] = bkey2i.get(str(tuple(int(v) for v in btr)), 0) if btr[0] >= 0 else 0
            ids[k, 2] = cam_id(yaw[min(s1, NF - 1)] - yaw[s0],
                               pitch[min(s1, NF - 1)] - pitch[s0], yt, pt)
        d["ww_action_ids"] = torch.from_numpy(ids)
        torch.save(d, cond_path)
        return 1
    except Exception as e:
        return f"{os.path.basename(cond_path)}: {type(e).__name__}: {str(e)[:80]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precomp", required=True)
    ap.add_argument("--yaw_thresh", type=float, default=1.2)
    ap.add_argument("--pitch_thresh", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    blob = torch.load(TABLES, map_location="cpu", weights_only=True)
    _G["pkey2i"] = {k: i for i, k in enumerate(blob["player_keys"])}
    _G["bkey2i"] = {k: i for i, k in enumerate(blob["boss_keys"])}

    rows = [json.loads(l) for l in open(f"{PILOT}/manifest.jsonl")]
    jobs = []
    miss = 0
    for r in rows:
        cond = os.path.join(args.precomp, "conditions",
                            r["video"].lstrip("/")).replace(".mp4", ".pt")
        state = r["video"].replace("/clips/", "/states/").replace(".mp4", ".npz")
        if os.path.exists(cond) and os.path.exists(state):
            jobs.append((cond, state, args.yaw_thresh, args.pitch_thresh))
        else:
            miss += 1
    print(f"[sidecar] {len(jobs)} jobs, {miss} missing", flush=True)
    n_ok, errs = 0, []
    with Pool(args.workers, initializer=lambda g: _G.update(g), initargs=(_G,)) as p:
        for i, r in enumerate(p.imap_unordered(process_one, jobs, chunksize=64)):
            if r == 1:
                n_ok += 1
            elif isinstance(r, str):
                errs.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)} injected={n_ok} errs={len(errs)}", flush=True)
    print(f"SIDECAR_DONE injected={n_ok} errs={len(errs)}", flush=True)
    for e in errs[:10]:
        print("  ERR", e, flush=True)


if __name__ == "__main__":
    main()
