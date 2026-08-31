"""复合句候选搜索:融合式措辞(非拼接),中心化余弦选优。"""
import itertools, torch
import torch.nn.functional as Fn

CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
GEMMA = "/data/yuzhewu/ltxwm/gemma-3-12b-it"
SEG = 32
BASE = {  # 已定稿基句(参与度量但不再选)
 "m_idle": "the character stands completely still, idle, not moving at all.",
 "m_W": "forward dash: the character races ahead into the depth of the world.",
 "m_S": "the character gives ground, easing back toward the viewer.",
 "m_A": "the character glides left across the ground.",
 "m_D": "the character breaks right, swinging wide to the right-hand side.",
 "v_idle": "the camera holds perfectly steady with no rotation.",
 "v_J": "panning left: the whole view sweeps leftward.",
 "v_L": "a rightward turn of the lens, the world sliding across to the right.",
 "v_I": "tilting skyward: the view climbs toward the heavens.",
 "v_K": "the camera dips low, the gaze falling to the ground below.",
}
CAND = {
 "m_WA": ["the character charges on a leftward diagonal, cutting ahead-left.",
          "surging to the front-left, the character angles across the open ground.",
          "an oblique advance: the character bears ahead and to the left."],
 "m_WD": ["racing diagonally rightward, the character darts toward the far right.",
          "the character storms ahead-right, veering out along the right diagonal.",
          "a slanting run to the right: the character pushes deep and rightward."],
 "m_SA": ["the character falls back toward the rear-left corner.",
          "drifting rear-left, the character slips backward and aside.",
          "a backward slide to the left: the character withdraws left-and-back."],
 "m_SD": ["retreating along the right diagonal, the character sinks to the rear-right.",
          "the character peels away backward-right, fading toward the lower right.",
          "a rearward drift rightward: the character backs off to the right corner."],
 "v_JI": ["the camera arcs up and to the left, climbing as it sweeps that way.",
          "rising leftward: the view lifts toward the upper-left sky.",
          "the lens curls left-and-up, the horizon dropping on the left."],
 "v_JK": ["the view plunges down while swinging to the left.",
          "sinking leftward: the camera noses down toward the lower-left ground.",
          "the lens tips left-and-down, terrain rising into the left of frame."],
 "v_LI": ["the lens soars to the upper right, lifting over the right horizon.",
          "climbing rightward: the view ascends toward the upper-right sky.",
          "the camera sweeps right while craning upward into the clouds."],
 "v_LK": ["the camera rolls down to the right, settling toward the lower-right ground.",
          "descending rightward: the view drops toward the bottom-right terrain.",
          "the lens leans right-and-down, the ground swelling at the lower right."],
}

def main():
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.blocks import PromptEncoder
    mp = ModelPaths.from_monolith(checkpoint_path=CKPT, gemma_root=GEMMA)
    enc = PromptEncoder(mp, dtype=torch.bfloat16, device=torch.device("cuda:0"))
    names, sents = [], []
    for k, s in BASE.items(): names.append((k, -1)); sents.append(s)
    for k, lst in CAND.items():
        for i, s in enumerate(lst): names.append((k, i)); sents.append(s)
    outs = []
    for s in sents:
        with torch.inference_mode(): outs.append(enc([s])[0])
        torch.cuda.empty_cache()
    embs = torch.stack([o.video_encoding.squeeze(0)[:SEG].float().mean(0) for o in outs])
    cen = Fn.normalize(embs - embs.mean(0, keepdim=True), dim=-1)
    E = {nm: cen[i] for i, nm in enumerate(names)}
    fixed = {k: E[(k, -1)] for k in BASE}
    # 贪心:按镜像对顺序选,目标=|与镜像伙伴| + 0.5*max|与同族已选/基句|
    order = [("m_WA","m_WD"),("m_SA","m_SD"),("v_JI","v_JK"),("v_LI","v_LK")]
    chosen = dict(fixed)
    pick = {}
    for a, b in order:
        best = None
        for ia, ib in itertools.product(range(3), range(3)):
            ea, eb = E[(a, ia)], E[(b, ib)]
            fam = [v for k, v in chosen.items() if k[:2] == a[:2]]
            pen = max(max(abs(float(ea @ f)) for f in fam), max(abs(float(eb @ f)) for f in fam))
            score = abs(float(ea @ eb)) + 0.5 * pen
            if best is None or score < best[0]: best = (score, ia, ib)
        pick[a], pick[b] = best[1], best[2]
        chosen[a], chosen[b] = E[(a, best[1])], E[(b, best[2])]
        print(f"{a}-{b}: score {best[0]:.3f} -> {a}#{best[1]} {b}#{best[2]}")
    print("\n== 定稿复合句 ==")
    for k, i in pick.items(): print(f'    "{k[2:]}": "{CAND[k][i]}",')
    ks = list(chosen)
    cos = torch.stack([chosen[k] for k in ks]) @ torch.stack([chosen[k] for k in ks]).T
    worst = sorted(((float(cos[i,j]), ks[i], ks[j]) for i in range(18) for j in range(i+1,18)), reverse=True)
    print("\n最高 10 对:")
    for c,a,b in worst[:10]: print(f"  {a:7s}-{b:7s}: {c:.3f}")

main()
