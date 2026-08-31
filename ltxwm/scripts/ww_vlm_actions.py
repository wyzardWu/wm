"""WildWorld 动作语义映射:Qwen2.5-VL 看代表段帧 + GT 状态,给每个动作三元组造句。

沿用 WildWorld 论文自己的配方(VLM + GT state 入 prompt)。
- player 键 (weapon_id, motion_bank_id, motion_id);monster 键 (type_id, bank, motion)
- 按总帧数降序处理(头部类先完成),增量写盘,可随时中断/续跑
- 输出 {key: {label, sentence, confidence, frames, segs_used}}

Usage:
  CUDA_VISIBLE_DEVICES=4 python ww_vlm_actions.py --entity player
  CUDA_VISIBLE_DEVICES=6 python ww_vlm_actions.py --entity monster
"""
import argparse
import ast
import json
import os
import re

import torch
from PIL import Image

WW = "/data/zijunlin/WildWorld/data_part1"
CENSUS = "/data/yuzhewu/ltxwm/ww_action_census.json"
QWEN = "/data/yuzhewu/ltxwm/qwen-vl"
WIKI_MON = "/data/yuzhewu/ltxwm/mhwilds_wiki/Monster-IDs.md"


def monster_names():
    names = {}
    for line in open(WIKI_MON):
        cols = [c.strip() for c in line.split("|")]
        if len(cols) > 6 and re.fullmatch(r"\\?-?\d+", cols[2].replace("\\", "")):
            names[int(cols[2].replace("\\", ""))] = cols[5]
    return names


def seg_frames(sid, start, length, n):
    import imageio.v3 as iio
    path = os.path.join(WW, sid, "rgb.mp4")
    idxs = [start + round(i * (length - 1) / (n - 1)) for i in range(n)]
    out = []
    for i in idxs:
        arr = iio.imread(path, index=i, plugin="pyav")
        out.append(Image.fromarray(arr).resize((640, 360)))
    return out


def build_query(entity, key, segs, mon_names, frames_per_seg):
    # 只用最长的一段,集中注意力读一段连续动作
    sid, start, length = max(segs, key=lambda s: s[2])
    images = seg_frames(sid, start, length, frames_per_seg)
    if entity == "player":
        subject = "the player character (the human hunter)"
        state = "Identify the weapon the hunter is holding from the frames."
        start_word = "the hunter"
    else:
        name = mon_names.get(key[0], f"type {key[0]}")
        subject = f"the monster ({name})"
        state = ""
        start_word = f"the {name}"
    text = (
        f"These {frames_per_seg} frames are consecutive moments, in temporal order, of "
        f"one animation of {subject} in Monster Hunter Wilds. {state}\n"
        f"Step 1 — under the heading OBSERVATIONS, describe in 2-3 short lines how "
        f"{subject}'s body actually changes from frame to frame: limbs, stance, "
        "which way it faces or moves, what the "
        + ("weapon" if entity == "player" else "head/tail/wings")
        + " does. Only report what you can see change.\n"
        "Step 2 — under the heading ANSWER, output JSON:\n"
        '{"label": "<2-6 word specific action name>", '
        f'"sentence": "<one 12-25 word present-tense sentence starting with '
        f"'{start_word}', describing this specific motion>\", "
        '"confidence": "<high|medium|low>"}\n'
        "The sentence must be concrete enough to tell this animation apart from other "
        "animations of the same character. Generic wordings like \"attacks the "
        "monster\" or \"swings the weapon\" are not acceptable. Ignore camera motion, "
        "HUD and scenery; mention the opponent only as a direction."
    )
    return images, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True, choices=["player", "monster"])
    ap.add_argument("--min_frames", type=int, default=10000)
    ap.add_argument("--max_segs", type=int, default=3)
    ap.add_argument("--frames_per_seg", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"/data/yuzhewu/ltxwm/ww_action_sentences_{args.entity}.json"

    census = json.load(open(CENSUS))[args.entity]
    todo = [(ast.literal_eval(k), v[0], v[1]) for k, v in census.items()
            if v[0] >= args.min_frames]
    todo.sort(key=lambda t: -t[1])
    done = json.load(open(out_path)) if os.path.exists(out_path) else {}
    print(f"[ww_vlm] {args.entity}: {len(todo)} classes >= {args.min_frames} frames, "
          f"{len(done)} already done", flush=True)
    mon_names = monster_names()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    proc = AutoProcessor.from_pretrained(QWEN)

    for n, (key, frames, segs) in enumerate(todo):
        kstr = str(key)
        if kstr in done:
            continue
        segs = segs[:args.max_segs]
        if not segs:
            done[kstr] = {"label": None, "sentence": None, "confidence": None,
                          "frames": frames, "segs_used": 0, "note": "no segments"}
            json.dump(done, open(out_path, "w"), indent=1)
            continue
        try:
            images, text = build_query(args.entity, key, segs, mon_names,
                                       args.frames_per_seg)
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
            msgs = [{"role": "user", "content": content}]
            chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[chat], images=images, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=320, do_sample=False)
            reply = proc.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]
            m = re.search(r"\{.*\}", reply, re.S)
            rec = json.loads(m.group(0)) if m else {"label": None, "sentence": reply.strip(),
                                                    "confidence": "low"}
            rec.update(frames=frames, segs_used=len(segs))
            done[kstr] = rec
        except Exception as e:
            done[kstr] = {"label": None, "sentence": None, "confidence": None,
                          "frames": frames, "segs_used": len(segs),
                          "note": f"error: {type(e).__name__}: {str(e)[:120]}"}
        json.dump(done, open(out_path, "w"), indent=1)
        if (n + 1) % 20 == 0 or n < 3:
            r = done[kstr]
            print(f"[{n + 1}/{len(todo)}] {kstr} ({frames}f) -> "
                  f"{r.get('label')} | {str(r.get('sentence'))[:80]}", flush=True)
    print("WW_VLM_DONE", args.entity, flush=True)


if __name__ == "__main__":
    main()
