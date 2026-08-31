"""Build LTX action sentence tables via the OFFICIAL text pipeline.

Encodes 9 movement + 9 view sentences through PromptEncoder (Gemma3 ->
EmbeddingsProcessor), exactly the path LTX-2.3 pretraining used, so the
teacher-student conditioning-consistency rule holds on the new base.

Output: /data/yuzhewu/ltxwm/tables/ltx_action_tables.pt
  { move_table: [9, 16, D], view_table: [9, 16, D],
    move_sentences, view_sentences, seg_len: 16 }
Rows are zero-padded/truncated to 16 tokens (true lengths recorded).

Usage (needs one GPU with ~30G free for Gemma):
  python build_ltx_action_tables.py [--device cuda:0]
"""
import argparse
import torch

CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
GEMMA = "/data/yuzhewu/ltxwm/gemma-3-12b-it"
SEG = 32

MOVE = {
    "":   "the character stands completely still, idle, not moving at all.",
    "W":  "the character runs forward, advancing ahead deeper into the scene.",
    "S":  "the character backs up, retreating backward toward the viewer.",
    "A":  "leftward motion: the character sidesteps left, drifting toward the left edge of the screen.",
    "D":  "rightward motion: the character sidesteps right, drifting toward the right edge of the screen.",
    "WA": "diagonal leftward advance: the character runs forward and left, angling toward the left side.",
    "WD": "diagonal rightward advance: the character runs forward and right, angling toward the right side.",
    "SA": "diagonal leftward retreat: the character backs up toward the left, retreating to the left side.",
    "SD": "diagonal rightward retreat: the character backs up toward the right, retreating to the right side.",
}
VIEW = {
    "":   "the camera holds perfectly steady with no rotation.",
    "J":  "leftward pan: the camera rotates left, the view sweeping toward the left.",
    "L":  "rightward pan: the camera rotates right, the view sweeping toward the right.",
    "I":  "upward tilt: the camera pitches up, the view rising toward the sky.",
    "K":  "downward tilt: the camera pitches down, the view dropping toward the ground.",
    "JI": "leftward and upward: the camera pans left while tilting up toward the sky.",
    "JK": "leftward and downward: the camera pans left while tilting down toward the ground.",
    "LI": "rightward and upward: the camera pans right while tilting up toward the sky.",
    "LK": "rightward and downward: the camera pans right while tilting down toward the ground.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/data/yuzhewu/ltxwm/tables/ltx_action_tables.pt")
    args = ap.parse_args()

    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.blocks import PromptEncoder

    mp = ModelPaths.from_monolith(checkpoint_path=CKPT, gemma_root=GEMMA)
    enc = PromptEncoder(mp, dtype=torch.bfloat16, device=torch.device(args.device))

    sents = list(MOVE.values()) + list(VIEW.values())
    outs = enc(sents)

    def to_row(o):
        # Connector output is seq-to-seq with padding slots replaced by
        # learnable registers (mask all-ones, fixed length). The first
        # positions stay aligned with the sentence tokens, so the first SEG
        # positions = real tokens + a few registers.
        emb = o.video_encoding.squeeze(0)          # [L_fixed, D]
        row = emb[:SEG].to(torch.bfloat16).clone()
        return row, int(emb.shape[0])

    rows = [to_row(o) for o in outs]
    lens = [n for _, n in rows]
    print("connector output lengths:", set(lens))

    mt = torch.stack([r for r, _ in rows[:9]])
    vt = torch.stack([r for r, _ in rows[9:]])
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "move_table": mt, "view_table": vt,
        "move_sentences": list(MOVE.values()), "view_sentences": list(VIEW.values()),
        "move_keys": list(MOVE.keys()), "view_keys": list(VIEW.keys()),
        "seg_len": SEG, "lens": lens,
    }, args.out)
    print("saved", args.out, "move", tuple(mt.shape), "view", tuple(vt.shape))


if __name__ == "__main__":
    main()
