"""Build v3 action tables: 措辞手术版(镜像对去相关,中心化余弦选优)。

基句由 search_v3_wording.py 实测选出(镜像对中心化余弦 ~0);斜向/复合句由
选定词族拼接。输出 ltx_action_tables_v31.pt,并打印全 18x18 中心化余弦矩阵。

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
    "W":  "the character walks onward, away from the camera, deeper into the scene.",
    "S":  "walking in reverse, the character approaches the camera.",
    "A":  "the character walks to the left side of the screen.",
    "D":  "going rightward, the character crosses over to the right of the view.",
    "WA": "the character goes forward-left, angling away toward the left.",
    "WD": "moving diagonally, the character heads deeper and to the right.",
    "SA": "the character moves backward-left, approaching the camera at an angle.",
    "SD": "the character steps back and to the right, nearing the viewer on the right diagonal.",
}
VIEW = {
    "":   "the camera holds perfectly steady with no rotation.",
    "J":  "the camera turns toward the left, and the view shifts that way.",
    "L":  "panning to the right, fresh scenery enters from the right side.",
    "I":  "looking upward, the camera raises its line of sight.",
    "K":  "the camera tilts downward, and more of the ground comes into view.",
    "JI": "the viewpoint rotates leftward as it rises toward the sky.",
    "JK": "turning to the left and looking down, the camera drops its gaze low and leftward.",
    "LI": "the camera pans right while angling upward.",
    "LK": "the viewpoint swings rightward as it lowers toward the ground.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/data/yuzhewu/ltxwm/tables/ltx_action_tables_v31.pt")
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
                "seg_len": SEG, "version": "v3.1_neutral_manner"}, args.out)
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
