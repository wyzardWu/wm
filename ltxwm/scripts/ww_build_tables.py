"""WildWorld 三槽动作表:player/boss 句子(VLM 表)+ 相机(复用 v3.1 view 表)。

每槽 32 token(与 v3.1 一致),帧上下文 = [caption 1024 | player 32 | boss 32 | cam 32]。
索引 0 = 空/兜底句(玩家 idle / 无怪物);尾部类和缺句类落到索引 0。
输出 ww_action_tables_v1.pt + 各族中心化余弦最差对(验收报告)。

Usage: CUDA_VISIBLE_DEVICES=0 python ww_build_tables.py
"""
import json

import torch

CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
GEMMA = "/data/yuzhewu/ltxwm/gemma-3-12b-it"
SEG = 32
NULL_PLAYER = "the hunter stands idle, doing nothing."
NULL_BOSS = "no monster is nearby."


def load_sentences(entity):
    d = json.load(open(f"/data/yuzhewu/ltxwm/ww_action_sentences_{entity}.json"))
    keys, sents = [], []
    for k, v in d.items():
        if v.get("sentence"):
            keys.append(k)
            sents.append(v["sentence"])
    return keys, sents


def main():
    pk, ps = load_sentences("player")
    mk, ms = load_sentences("monster")
    print(f"player {len(ps)} + boss {len(ms)} sentences", flush=True)

    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.blocks import PromptEncoder
    mp = ModelPaths.from_monolith(checkpoint_path=CKPT, gemma_root=GEMMA)
    enc = PromptEncoder(mp, dtype=torch.bfloat16, device=torch.device("cuda:0"))

    def encode_all(sents):
        rows = []
        for i, s in enumerate(sents):
            with torch.inference_mode():
                o = enc([s])[0]
            rows.append(o.video_encoding.squeeze(0)[:SEG].to(torch.bfloat16).cpu().clone())
            torch.cuda.empty_cache()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(sents)}", flush=True)
        return torch.stack(rows)

    pt = encode_all([NULL_PLAYER] + ps)          # [1+P, 32, D]
    bt = encode_all([NULL_BOSS] + ms)            # [1+B, 32, D]

    v31 = torch.load("/data/yuzhewu/ltxwm/tables/ltx_action_tables_v31.pt",
                     map_location="cpu", weights_only=True)
    ct = v31["view_table"]                       # [9, 32, D] idle,J,L,I,K,JI,JK,LI,LK

    torch.save({"player_table": pt, "boss_table": bt, "cam_table": ct,
                "player_keys": ["NULL"] + pk, "boss_keys": ["NULL"] + mk,
                "cam_keys": ["", "J", "L", "I", "K", "JI", "JK", "LI", "LK"],
                "seg_len": SEG, "version": "ww_v1"},
               "/data/yuzhewu/ltxwm/tables/ww_action_tables_v1.pt")
    print("saved ww_action_tables_v1.pt", pt.shape, bt.shape, ct.shape, flush=True)

    # 验收:族内中心化余弦最差对
    import torch.nn.functional as Fn
    for name, tab, keys in [("player", pt, ["NULL"] + pk), ("boss", bt, ["NULL"] + mk)]:
        e = tab.float().mean(1)
        c = Fn.normalize(e - e.mean(0, keepdim=True), dim=-1)
        cos = c @ c.T
        cos.fill_diagonal_(-2)
        vals, idx = cos.flatten().topk(10)
        print(f"\n{name} 最高余弦对(可能需要聚类合并):")
        n = cos.shape[0]
        seen = set()
        for v, ij in zip(vals.tolist(), idx.tolist()):
            i, j = ij // n, ij % n
            if (j, i) in seen:
                continue
            seen.add((i, j))
            print(f"  {v:.3f}  {keys[i]}  <->  {keys[j]}")
    print("WW_TABLES_DONE", flush=True)


if __name__ == "__main__":
    main()
