"""Build v3 action tables: 措辞手术版(镜像对去相关,中心化余弦选优)。

基句由 search_v3_wording.py 实测选出(镜像对中心化余弦 ~0);斜向/复合句由
选定词族拼接。输出 ltx_action_tables_v3.pt,并打印全 18x18 中心化余弦矩阵。

Usage: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=<g> \
       python build_ltx_action_tables_v3.py
"""
import argparse
import torch

CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
GEMMA = "/data/yuzhewu/ltxwm/gemma-3-12b-it"
SEG = 32

MOVE = {
    "":   "the character stands completely still, idle, not moving at all.",
    "W":  "forward dash: the character races ahead into the depth of the world.",
    "S":  "the character gives ground, easing back toward the viewer.",
    "A":  "the character glides left across the ground.",
    "D":  "the character breaks right, swinging wide to the right-hand side.",
    "WA": "an oblique advance: the character bears ahead and to the left.",
    "WD": "racing diagonally rightward, the character darts toward the far right.",
    "SA": "drifting rear-left, the character slips backward and aside.",
    "SD": "retreating along the right diagonal, the character sinks to the rear-right.",
}
VIEW = {
    "":   "the camera holds perfectly steady with no rotation.",
    "J":  "panning left: the whole view sweeps leftward.",
    "L":  "a rightward turn of the lens, the world sliding across to the right.",
    "I":  "tilting skyward: the view climbs toward the heavens.",
    "K":  "the camera dips low, the gaze falling to the ground below.",
    "JI": "the camera arcs up and to the left, climbing as it sweeps that way.",
    "JK": "the view plunges down while swinging to the left.",
    "LI": "the lens soars to the upper right, lifting over the right horizon.",
    "LK": "descending rightward: the view drops toward the bottom-right terrain.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/data/yuzhewu/ltxwm/tables/ltx_action_tables_v3.pt")
    args = ap.parse_args()

    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.blocks import PromptEncoder

    mp = ModelPaths.from_monolith(checkpoint_path=CKPT, gemma_root=GEMMA)
    enc = PromptEncoder(mp, dtype=torch.bfloat16, device=torch.device(args.device))

    sents = list(MOVE.values()) + list(VIEW.values())
    outs = []
    for s in sents:
        with torch.inference_mode():
            outs.append(enc([s])[0])
        torch.cuda.empty_cache()

    rows = [o.video_encoding.squeeze(0)[:SEG].to(torch.bfloat16).clone() for o in outs]
    mt = torch.stack(rows[:9])
    vt = torch.stack(rows[9:])
    torch.save({"move_table": mt, "view_table": vt,
                "move_sentences": list(MOVE.values()), "view_sentences": list(VIEW.values()),
                "seg_len": SEG, "version": "v3_wording_surgery"}, args.out)
    print("saved", args.out, mt.shape, vt.shape)

    # 验收:全 18x18 中心化余弦
    import torch.nn.functional as Fn
    emb = torch.cat([mt, vt]).float().mean(1)
    cen = Fn.normalize(emb - emb.mean(0, keepdim=True), dim=-1)
    names = ["m_" + (k or "idle") for k in MOVE] + ["v_" + (k or "idle") for k in VIEW]
    cos = cen @ cen.T
    worst = []
    for i in range(18):
        for j in range(i + 1, 18):
            worst.append((float(cos[i, j]), names[i], names[j]))
    worst.sort(reverse=True)
    print("\n最高的 12 对(危险区):")
    for c, a, b in worst[:12]:
        print(f"  {a:8s}-{b:8s}: {c:.3f}")
    print("\n镜像对:")
    for a, b in [("m_W","m_S"),("m_A","m_D"),("v_J","v_L"),("v_I","v_K"),
                 ("m_WA","m_WD"),("m_SA","m_SD"),("v_JI","v_JK"),("v_LI","v_LK")]:
        print(f"  {a}-{b}: {float(cos[names.index(a), names.index(b)]):.3f}")

if __name__ == "__main__":
    main()
