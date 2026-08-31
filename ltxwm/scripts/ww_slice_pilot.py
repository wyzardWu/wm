"""WildWorld pilot 切片:6s 窗(180帧@30fps),战斗窗+漫游窗,ffmpeg 精确切,
每 clip 存 state 切片 npz(动作 sidecar 原料) + metadata.jsonl 行。

窗口判定(与上周验证的 ww_roam/find_combat 逻辑一致):
- combat: 怪物在场且平均距离 < 15
- roam:   移动 bank 占比 >= 0.6 且 (无怪物 或 距离 > 40)
每样本配额 combat<=15, roam<=5;可中断续跑(按 sample done 标记)。

Usage: python ww_slice_pilot.py --out /data/yuzhewu/ltxwm/data/ww_pilot [--workers 24]
"""
import argparse
import ast
import glob
import json
import os
import subprocess
from multiprocessing import Pool

import numpy as np
import pandas as pd

WW = "/data/zijunlin/WildWorld/data_part1"
MOVE_BANKS = {50, 103, 51, 14, 7}
WIN, STRIDE, FPS = 180, 90, 30
QUOTA = {"combat": 15, "roam": 5}

COLS = ["npc.list.1.weapon_id", "npc.list.1.motion_bank_id", "npc.list.1.motion_id",
        "npc.list.1.motion_frame", "npc.list.1.pos", "npc.list.1.rot",
        "monster.list.1.type_id", "monster.list.1.motion_bank_id",
        "monster.list.1.motion_id", "monster.list.1.motion_frame",
        "monster.list.1.pos", "camera.K", "camera.pos", "camera.rot"]


def parse_vec(series, dim):
    out = np.full((len(series), dim), np.nan, dtype=np.float32)
    for i, s in enumerate(series):
        if isinstance(s, str):
            try:
                out[i] = ast.literal_eval(s)
            except Exception:
                pass
    return out


def process_sample(args_tuple):
    sid, out_root = args_tuple
    done_flag = os.path.join(out_root, "done", sid)
    if os.path.exists(done_flag):
        return sid, 0, "done"
    csv = os.path.join(WW, sid, "state.csv")
    mp4 = os.path.join(WW, sid, "rgb.mp4")
    if not (os.path.exists(csv) and os.path.exists(mp4)):
        return sid, 0, "missing"
    try:
        df = pd.read_csv(csv, usecols=lambda c: c in COLS)
    except Exception as e:
        return sid, 0, f"csv_err:{type(e).__name__}"
    n = len(df)
    bank = df["npc.list.1.motion_bank_id"].fillna(-1).astype(int).values
    ppos = parse_vec(df["npc.list.1.pos"], 3)
    mpos = parse_vec(df["monster.list.1.pos"], 3)
    dist = np.linalg.norm(ppos - mpos, axis=1)
    ismove = np.isin(bank, list(MOVE_BANKS)).astype(np.float32)

    wins = []
    for s0 in range(0, n - WIN, STRIDE):
        seg = slice(s0, s0 + WIN)
        d = np.nanmean(dist[seg])
        mf = float(ismove[seg].mean())
        if not np.isnan(d) and d < 15:
            wins.append((s0, "combat", d, mf))
        elif mf >= 0.6 and (np.isnan(d) or d > 40):
            wins.append((s0, "roam", d, mf))
    # 贪心去重叠:combat 按距离近优先,roam 按移动占比高优先
    picked, used = [], np.zeros(n, bool)
    for typ, keyfun in [("combat", lambda w: w[2]), ("roam", lambda w: -w[3])]:
        cnt = 0
        for w in sorted([w for w in wins if w[1] == typ], key=keyfun):
            if cnt >= QUOTA[typ]:
                break
            s0 = w[0]
            if used[s0:s0 + WIN].any():
                continue
            used[s0:s0 + WIN] = True
            picked.append(w)
            cnt += 1

    meta_rows = []
    for s0, typ, d, mf in picked:
        name = f"{sid}_{s0:06d}_{typ}"
        clip = os.path.join(out_root, "clips", f"{name}.mp4")
        npz = os.path.join(out_root, "states", f"{name}.npz")
        if not os.path.exists(clip):
            r = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-ss", f"{s0 / FPS:.4f}", "-i", mp4,
                 "-frames:v", str(WIN), "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "18", "-an", "-y", clip],
                capture_output=True)
            if r.returncode != 0:
                continue
        seg = df.iloc[s0:s0 + WIN]
        np.savez_compressed(
            npz,
            weapon=seg["npc.list.1.weapon_id"].fillna(-1).astype(int).values,
            p_bank=seg["npc.list.1.motion_bank_id"].fillna(-1).astype(int).values,
            p_motion=seg["npc.list.1.motion_id"].fillna(-1).astype(int).values,
            p_mframe=seg["npc.list.1.motion_frame"].fillna(-1).values.astype(np.float32),
            m_type=seg["monster.list.1.type_id"].fillna(-1).astype(int).values,
            m_bank=seg["monster.list.1.motion_bank_id"].fillna(-1).astype(int).values,
            m_motion=seg["monster.list.1.motion_id"].fillna(-1).astype(int).values,
            m_mframe=seg["monster.list.1.motion_frame"].fillna(-1).values.astype(np.float32),
            cam_pos=parse_vec(seg["camera.pos"], 3),
            cam_rot=parse_vec(seg["camera.rot"], 4),
            cam_K=parse_vec(seg["camera.K"], 4),
            dist=dist[s0:s0 + WIN].astype(np.float32))
        meta_rows.append({"clip": f"clips/{name}.mp4", "state": f"states/{name}.npz",
                          "sample": sid, "start": s0, "frames": WIN, "type": typ,
                          "mean_dist": None if np.isnan(d) else round(float(d), 1),
                          "move_frac": round(mf, 2)})
    with open(os.path.join(out_root, "meta_parts", f"{sid}.jsonl"), "w") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")
    open(done_flag, "w").close()
    return sid, len(meta_rows), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/yuzhewu/ltxwm/data/ww_pilot")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="only first N samples (0=all)")
    args = ap.parse_args()
    for sub in ["clips", "states", "meta_parts", "done"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    sids = sorted(os.path.basename(os.path.dirname(f))
                  for f in glob.glob(f"{WW}/*/state.csv"))
    if args.limit:
        sids = sids[:args.limit]
    print(f"[slice] {len(sids)} samples, workers={args.workers}", flush=True)
    tot = 0
    with Pool(args.workers) as pool:
        for i, (sid, k, st) in enumerate(
                pool.imap_unordered(process_sample, [(s, args.out) for s in sids])):
            tot += k
            if (i + 1) % 50 == 0 or st not in ("ok", "done"):
                print(f"[{i + 1}/{len(sids)}] {sid}: {k} clips ({st}), total {tot}",
                      flush=True)
    print(f"SLICE_DONE total_clips={tot}", flush=True)


if __name__ == "__main__":
    main()
