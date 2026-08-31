"""v3 词表措辞搜索:候选句过官方 PromptEncoder(Gemma+connector),
选出使镜像/兄弟对余弦最小的组合。度量层 = 模型实际看到的 post-connector 空间。

用法: CUDA_VISIBLE_DEVICES=5 python search_v3_wording.py
"""
import itertools
import torch

CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
GEMMA = "/data/yuzhewu/ltxwm/gemma-3-12b-it"
SEG = 32

# 候选池:镜像两侧刻意用不同词族/句式(V Rising v3 方法论)
CAND = {
    "W": ["the character sprints ahead, charging deeper into the distance.",
          "the character presses onward, plunging further into the scene.",
          "forward dash: the character races ahead into the depth of the world."],
    "S": ["the character withdraws, stepping back the way it came.",
          "the character gives ground, easing back toward the viewer.",
          "a slow retreat: the character backs away from the scene."],
    "A": ["the character slips sideways to the left, hugging the left flank.",
          "leftward slide: the character strafes toward the left edge.",
          "the character glides left across the ground."],
    "D": ["the character darts off to starboard, cutting across to the right.",
          "a rightward burst: the character veers hard right across the field.",
          "the character breaks right, swinging wide to the right-hand side."],
    "J": ["the camera swings toward the left horizon in a smooth arc.",
          "panning left: the whole view sweeps leftward.",
          "the viewpoint wheels to the left, revealing what lies leftward."],
    "L": ["the camera wheels around clockwise, the scenery streaming to starboard.",
          "a rightward turn of the lens, the world sliding across to the right.",
          "the view rotates rightward, new scenery entering from the right."],
    "I": ["the camera cranes upward, lifting the gaze into the sky.",
          "tilting skyward: the view climbs toward the heavens.",
          "the lens angles up, the horizon sinking as the sky fills the frame."],
    "K": ["the camera dips low, the gaze falling to the ground below.",
          "tilting earthward: the view descends toward the terrain.",
          "the lens bows down, the ground rising as the sky slips away."],
}
IDLE_MOVE = "the character stands completely still, idle, not moving at all."
IDLE_VIEW = "the camera holds perfectly steady with no rotation."

# 需要压低的对(镜像 + 同族兄弟)
CRITICAL = [("W","S"), ("A","D"), ("J","L"), ("I","K")]


def main():
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.blocks import PromptEncoder
    mp = ModelPaths.from_monolith(checkpoint_path=CKPT, gemma_root=GEMMA)
    enc = PromptEncoder(mp, dtype=torch.bfloat16, device=torch.device("cuda:0"))

    keys, sents = [], []
    for k, lst in CAND.items():
        for i, s in enumerate(lst):
            keys.append((k, i)); sents.append(s)
    sents += [IDLE_MOVE, IDLE_VIEW]

    outs = []
    for s in sents:
        with torch.inference_mode():
            outs.append(enc([s])[0])
        torch.cuda.empty_cache()
    embs = []
    for o in outs:
        e = o.video_encoding.squeeze(0)[:SEG].float().mean(0)
        embs.append(torch.nn.functional.normalize(e, dim=-1))
    embs = torch.stack(embs)
    torch.save({"keys": keys + [("idle_m",0),("idle_v",0)], "sents": sents, "embs": embs},
               "/data/yuzhewu/ltxwm/tables/v3_candidate_embs.pt")
    # 中心化:去掉全候选公共均值分量后再归一(去地板)
    centered = torch.nn.functional.normalize(embs - embs.mean(0, keepdim=True), dim=-1)
    cand_emb = {kk: centered[i] for i, kk in enumerate(keys)}

    # 穷举每个 critical 对两侧的候选组合,选余弦最小者;各对独立(键不重叠)
    choice = {}
    for a, b in CRITICAL:
        best = None
        for ia, ib in itertools.product(range(len(CAND[a])), range(len(CAND[b]))):
            c = float(cand_emb[(a, ia)] @ cand_emb[(b, ib)])
            if best is None or c < best[0]:
                best = (c, ia, ib)
        choice[a], choice[b] = best[1], best[2]
        print(f"{a}-{b}: best cos {best[0]:.3f}  ({a}#{best[1]} / {b}#{best[2]})")

    final = {k: CAND[k][i] for k, i in choice.items()}
    print("\n== 选定句 ==")
    for k, s in final.items():
        print(f"{k}: {s}")

    # 汇报全交叉矩阵(含 idle)
    sel = {k: cand_emb[(k, choice[k])] for k in CAND}
    sel["idle_m"] = centered[-2]; sel["idle_v"] = centered[-1]
    # v2 旧句的中心化基线(同空间同尺子)
    print("\n== v2 旧表在中心化空间的镜像对(对照) ==")
    old = torch.load('/data/yuzhewu/ltxwm/tables/ltx_action_tables.pt', map_location='cpu', weights_only=True)
    ov = torch.cat([old['move_table'].float().mean(1), old['view_table'].float().mean(1)])
    ovc = torch.nn.functional.normalize(ov - ov.mean(0, keepdim=True), dim=-1)
    MO = ["idle","W","S","A","D","WA","WD","SA","SD"]; VI=["idle","J","L","I","K","JI","JK","LI","LK"]
    for fam, lab in [(0,MO),(9,VI)]:
        for a,b in ([("W","S"),("A","D")] if fam==0 else [("J","L"),("I","K")]):
            print(f"  v2 {a}-{b}: {float(ovc[fam+lab.index(a)] @ ovc[fam+lab.index(b)]):.3f}")
    names = list(sel)
    print("\n== 选定组合全对余弦 ==")
    for i, a in enumerate(names):
        for b in names[i+1:]:
            c = float(sel[a] @ sel[b])
            flag = " <==" if c > 0.85 else ""
            print(f"{a:7s}-{b:7s}: {c:.3f}{flag}")

if __name__ == "__main__":
    main()
