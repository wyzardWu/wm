"""v3.1:全类速度中性措辞,去相关只靠句法/参照系差异。测中心化余弦。"""
import itertools, torch
import torch.nn.functional as Fn
CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
GEMMA = "/data/yuzhewu/ltxwm/gemma-3-12b-it"
SEG = 32
CAND = {
 "W": ["the character walks onward, away from the camera, deeper into the scene.",
       "moving off into the distance, the character grows farther from the viewer.",
       "the character heads straight into the depth of the world."],
 "S": ["the character steps backward, coming closer to the viewer.",
       "walking in reverse, the character approaches the camera.",
       "the character moves back the way it came, toward the near side."],
 "A": ["the character moves sideways toward the left of the frame.",
       "traveling leftward, the character crosses the view to the left.",
       "the character walks to the left side of the screen."],
 "D": ["the character travels toward the right edge of the picture.",
       "going rightward, the character crosses over to the right of the view.",
       "the character walks along to the right-hand part of the screen."],
 "J": ["the camera turns toward the left, and the view shifts that way.",
       "the viewpoint rotates leftward at a steady pace.",
       "panning to the left, the camera brings new scenery in from the left."],
 "L": ["the camera turns to the right, the picture moving along with it.",
       "the viewpoint swings rightward, steady and even.",
       "panning to the right, fresh scenery enters from the right side."],
 "I": ["the camera tilts upward, and more of the sky comes into view.",
       "the view angles up toward what lies above.",
       "looking upward, the camera raises its line of sight."],
 "K": ["the camera tilts downward, and more of the ground comes into view.",
       "the view angles down toward what lies below.",
       "looking downward, the camera lowers its line of sight."],
}
def main():
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.blocks import PromptEncoder
    mp = ModelPaths.from_monolith(checkpoint_path=CKPT, gemma_root=GEMMA)
    enc = PromptEncoder(mp, dtype=torch.bfloat16, device=torch.device("cuda:0"))
    names, sents = [], []
    for k, lst in CAND.items():
        for i, s in enumerate(lst): names.append((k,i)); sents.append(s)
    outs = []
    for s in sents:
        with torch.inference_mode(): outs.append(enc([s])[0])
        torch.cuda.empty_cache()
    embs = torch.stack([o.video_encoding.squeeze(0)[:SEG].float().mean(0) for o in outs])
    cen = Fn.normalize(embs - embs.mean(0, keepdim=True), dim=-1)
    E = {nm: cen[i] for i, nm in enumerate(names)}
    pick = {}
    for a, b in [("W","S"),("A","D"),("J","L"),("I","K")]:
        best = min(((abs(float(E[(a,ia)] @ E[(b,ib)])), ia, ib)
                    for ia, ib in itertools.product(range(3), range(3))))
        pick[a], pick[b] = best[1], best[2]
        print(f"{a}-{b}: best |cos| {best[0]:.3f} -> {a}#{best[1]} / {b}#{best[2]}")
    print()
    for k, i in pick.items(): print(f'  "{k}": "{CAND[k][i]}"')
    sel = {k: E[(k, pick[k])] for k in CAND}
    ks = list(sel)
    cos = torch.stack(list(sel.values())) @ torch.stack(list(sel.values())).T
    print("\n全对:")
    for i in range(8):
        for j in range(i+1,8):
            c=float(cos[i,j]); print(f"  {ks[i]}-{ks[j]}: {c:.3f}" + ("  <==" if abs(c)>0.5 else ""))
main()
