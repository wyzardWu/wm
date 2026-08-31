"""抽样叠印:训练用 121 帧 + 每帧原始按键 + 所属 latent 组的分箱标签,供人工检验."""
import os, sys
import numpy as np
import imageio.v3 as iio
import imageio
from PIL import Image, ImageDraw, ImageFont

MOVE_KEYS = ["idle", "W", "S", "A", "D", "WA", "WD", "SA", "SD"]
VIEW_KEYS = ["idle", "J", "L", "I", "K", "JI", "JK", "LI", "LK"]
KEYS11 = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Sp"]
GROUPS = [(0, 1)] + [(1 + 8 * g, 9 + 8 * g) for g in range(15)]

def group_of(f):
    for gi, (a, b) in enumerate(GROUPS):
        if a <= f < b:
            return gi
    return 15

def main(video, ids_path, out):
    raw = np.load(video.replace(".mp4", ".npy"))          # [130,17]
    ids = np.load(ids_path)                               # [16,2]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    w = imageio.get_writer(out, fps=24, codec="libx264", quality=7)
    for f, frame in enumerate(iio.imiter(video)):
        if f >= 121:
            break
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        pressed = "+".join(k for k, i in zip(KEYS11, range(11)) if raw[f, i] > 0.5) or "-"
        gi = group_of(f)
        mv, vw = MOVE_KEYS[ids[gi, 0]], VIEW_KEYS[ids[gi, 1]]
        lines = [f"f{f:03d} raw: {pressed}",
                 f"lat{gi:02d} move={mv} view={vw}"]
        d.rectangle([0, 0, 430, 74], fill=(0, 0, 0))
        for li, t in enumerate(lines):
            d.text((8, 6 + 32 * li), t, fill=(0, 255, 80) if li else (255, 255, 0), font=font)
        w.append_data(np.asarray(img))
    w.close()
    print("saved", out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
