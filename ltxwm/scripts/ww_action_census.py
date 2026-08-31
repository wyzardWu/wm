"""WildWorld 动作普查:玩家 (weapon,bank,motion) 三元组 + 怪物 (type,bank,motion),
频次/时长/代表片段(采样几个 sample+帧区间,供后续 VLM 看帧造句用)。"""
import ast, glob, json, os
import numpy as np, pandas as pd
from collections import defaultdict

out_p, out_m = defaultdict(lambda: [0, []]), defaultdict(lambda: [0, []])
files = sorted(glob.glob('/data/zijunlin/WildWorld/data_part1/*/state.csv'))
print("samples:", len(files))
for n, f in enumerate(files):
    sid = f.split('/')[-2]
    try:
        df = pd.read_csv(f, usecols=['npc.list.1.weapon_id','npc.list.1.motion_bank_id','npc.list.1.motion_id',
                                     'monster.list.1.type_id','monster.list.1.motion_bank_id','monster.list.1.motion_id'])
    except Exception as e:
        print("skip", sid, e); continue
    p = df[['npc.list.1.weapon_id','npc.list.1.motion_bank_id','npc.list.1.motion_id']].fillna(-1).astype(int)
    key = list(zip(p.iloc[:,0], p.iloc[:,1], p.iloc[:,2]))
    # 连续段:记 (sample, start, len),每三元组最多存 5 个代表段
    prev, start = None, 0
    for i, k in enumerate(key + [None]):
        if k != prev:
            if prev is not None and prev[2] >= 0:
                seg = i - start
                out_p[prev][0] += seg
                if len(out_p[prev][1]) < 5 and seg >= 15:
                    out_p[prev][1].append((sid, start, seg))
            prev, start = k, i
    m = df[['monster.list.1.type_id','monster.list.1.motion_bank_id','monster.list.1.motion_id']].fillna(-1).astype(int)
    keym = list(zip(m.iloc[:,0], m.iloc[:,1], m.iloc[:,2]))
    prev, start = None, 0
    for i, k in enumerate(keym + [None]):
        if k != prev:
            if prev is not None and prev[0] >= 0 and prev[2] >= 0:
                seg = i - start
                out_m[prev][0] += seg
                if len(out_m[prev][1]) < 5 and seg >= 15:
                    out_m[prev][1].append((sid, start, seg))
            prev, start = k, i
    if (n+1) % 200 == 0:
        print(f"{n+1}/{len(files)} player_triplets={len(out_p)} monster_triplets={len(out_m)}", flush=True)

json.dump({"player": {str(k): v for k, v in out_p.items()},
           "monster": {str(k): v for k, v in out_m.items()}},
          open('/data/yuzhewu/ltxwm/ww_action_census.json', 'w'))
tp = sorted(out_p.items(), key=lambda kv: -kv[1][0])
tm = sorted(out_m.items(), key=lambda kv: -kv[1][0])
print(f"\n玩家三元组总数 {len(out_p)},Top15:")
for k, v in tp[:15]: print(f"  wp{k[0]} bank{k[1]} motion{k[2]}: {v[0]} 帧")
print(f"\n怪物 (种类,bank,motion) 总数 {len(out_m)},Top10:")
for k, v in tm[:10]: print(f"  type{k[0]} bank{k[1]} motion{k[2]}: {v[0]} 帧")
print("CENSUS_DONE")
