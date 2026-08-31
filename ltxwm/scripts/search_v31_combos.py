"""v3.1 复合句:中性速度,融合式,含基句一起度量。"""
import itertools, torch
import torch.nn.functional as Fn
CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
GEMMA = "/data/yuzhewu/ltxwm/gemma-3-12b-it"
SEG = 32
BASE = {
 "m_idle": "the character stands completely still, idle, not moving at all.",
 "m_W": "the character walks onward, away from the camera, deeper into the scene.",
 "m_S": "walking in reverse, the character approaches the camera.",
 "m_A": "the character walks to the left side of the screen.",
 "m_D": "going rightward, the character crosses over to the right of the view.",
 "v_idle": "the camera holds perfectly steady with no rotation.",
 "v_J": "the viewpoint rotates leftward at a steady pace.",
 "v_L": "panning to the right, fresh scenery enters from the right side.",
 "v_I": "looking upward, the camera raises its line of sight.",
 "v_K": "the camera tilts downward, and more of the ground comes into view.",
}
CAND = {
 "m_WA": ["the character walks ahead and leftward, on a diagonal away from the camera.",
          "moving deeper and to the left, the character follows the left diagonal.",
          "the character goes forward-left, angling away toward the left."],
 "m_WD": ["moving diagonally, the character heads deeper and to the right.",
          "the character walks ahead and rightward along the right diagonal.",
          "the character goes forward-right, angling away toward the right."],
 "m_SA": ["the character steps back and to the left, nearing the viewer on the left diagonal.",
          "walking in reverse toward the left, the character comes closer on that side.",
          "the character moves backward-left, approaching the camera at an angle."],
 "m_SD": ["the character steps back and to the right, nearing the viewer on the right diagonal.",
          "walking in reverse toward the right, the character comes closer on that side.",
          "the character moves backward-right, approaching the camera at an angle."],
 "v_JI": ["the camera turns left while angling upward.",
          "the viewpoint rotates leftward as it rises toward the sky.",
          "turning to the left and looking up, the camera shifts its gaze high and leftward."],
 "v_JK": ["the camera turns left while angling downward.",
          "the viewpoint rotates leftward as it lowers toward the ground.",
          "turning to the left and looking down, the camera drops its gaze low and leftward."],
 "v_LI": ["the camera pans right while angling upward.",
          "the viewpoint swings rightward as it rises toward the sky.",
          "turning to the right and looking up, the camera lifts its gaze high and rightward."],
 "v_LK": ["the camera pans right while angling downward.",
          "the viewpoint swings rightward as it lowers toward the ground.",
          "turning to the right and looking down, the camera drops its gaze low and rightward."],
}
def main():
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.blocks import PromptEncoder
    mp = ModelPaths.from_monolith(checkpoint_path=CKPT, gemma_root=GEMMA)
    enc = PromptEncoder(mp, dtype=torch.bfloat16, device=torch.device("cuda:0"))
    names, sents = [], []
    for k, s in BASE.items(): names.append((k,-1)); sents.append(s)
    for k, lst in CAND.items():
        for i, s in enumerate(lst): names.append((k,i)); sents.append(s)
    outs = []
    for s in sents:
        with torch.inference_mode(): outs.append(enc([s])[0])
        torch.cuda.empty_cache()
    embs = torch.stack([o.video_encoding.squeeze(0)[:SEG].float().mean(0) for o in outs])
    cen = Fn.normalize(embs - embs.mean(0, keepdim=True), dim=-1)
    E = {nm: cen[i] for i, nm in enumerate(names)}
    chosen = {k: E[(k,-1)] for k in BASE}
    pick = {}
    for a, b in [("m_WA","m_WD"),("m_SA","m_SD"),("v_JI","v_JK"),("v_LI","v_LK")]:
        best = None
        for ia, ib in itertools.product(range(3), range(3)):
            ea, eb = E[(a,ia)], E[(b,ib)]
            fam = [v for k, v in chosen.items() if k[:2] == a[:2]]
            pen = max(max(float(ea @ f) for f in fam), max(float(eb @ f) for f in fam))  # 只罚正相关
            score = abs(float(ea @ eb)) + 0.5 * max(pen, 0)
            if best is None or score < best[0]: best = (score, ia, ib)
        pick[a], pick[b] = best[1], best[2]
        chosen[a], chosen[b] = E[(a,best[1])], E[(b,best[2])]
        print(f"{a}-{b}: score {best[0]:.3f} -> #{best[1]}/#{best[2]}")
    print()
    for k, i in pick.items(): print(f'  "{k[2:]}": "{CAND[k][i]}",')
    ks = list(chosen)
    M = torch.stack(list(chosen.values()))
    cos = M @ M.T
    worst = sorted(((float(cos[i,j]), ks[i], ks[j]) for i in range(18) for j in range(i+1,18)), reverse=True)
    print("\n最高正相关 8 对:")
    for c,a,b in worst[:8]: print(f"  {a:7s}-{b:7s}: {c:.3f}")
main()
