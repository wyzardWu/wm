"""
MP4 IO for GameMaster rollout/eval.

PyAV (`av`) is the ONLY video writer present in the gamemaster conda env — imageio,
imageio_ffmpeg, opencv (cv2) and torchvision are all absent (verified). So we encode
h264/yuv420p via PyAV. yuv420p needs even width/height (480x832 is fine). Tensors are
the model/VAE convention: float [3, T, H, W] in [-1, 1].
"""

import json
import os
import re
import numpy as np
import torch


def chw_to_uint8(pix):
    """[3, T, H, W] float in [-1, 1]  ->  uint8 numpy [T, H, W, 3] (RGB).

    Matches Incantation inference.py: ((x.clamp(-1,1)+1)/2*255).byte().
    """
    if not isinstance(pix, torch.Tensor):
        pix = torch.as_tensor(pix)
    x = pix.detach().float().cpu().clamp(-1, 1)
    assert x.dim() == 4 and x.shape[0] == 3, f"expect [3,T,H,W], got {tuple(x.shape)}"
    x = ((x + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
    return x.permute(1, 2, 3, 0).contiguous().numpy()      # [T, H, W, 3]


def write_mp4(frames, path, fps=16, crf=17, codec="libx264"):
    """Write uint8 frames [T, H, W, 3] (RGB) to an h264 MP4 via PyAV.

    `frames` may be a uint8 numpy array, or a float [3,T,H,W] tensor in [-1,1]
    (auto-converted). H and W must be even (yuv420p). Returns `path`.
    """
    import av

    if isinstance(frames, torch.Tensor) or (isinstance(frames, np.ndarray) and frames.ndim == 4
                                            and frames.shape[0] == 3 and frames.dtype != np.uint8):
        frames = chw_to_uint8(frames)
    frames = np.asarray(frames)
    assert frames.ndim == 4 and frames.shape[-1] == 3 and frames.dtype == np.uint8, \
        f"expect uint8 [T,H,W,3], got {frames.shape} {frames.dtype}"
    T, H, W, _ = frames.shape
    assert H % 2 == 0 and W % 2 == 0, f"h264 yuv420p needs even H,W (got {H}x{W})"

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream(codec, rate=int(fps))
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        for t in range(T):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames[t]), format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():            # flush
            container.mux(pkt)
    finally:
        container.close()
    return path


def save_png(pix, path, frame=0):
    """Save one frame of a [3,T,H,W] (or [3,H,W]) float clip in [-1,1] as a PNG."""
    from PIL import Image
    f = pix if pix.dim() == 3 else pix[:, frame]
    x = f.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(x).save(path)
    return path


def _slug(label, i):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(label)).strip("_")
    return s or f"p{i}"


def save_eval_bundle(out_dir, panels, fps=16, start_frame=None, meta=None):
    """Write a SELF-DOCUMENTING eval bundle to out_dir:
        panel_{i:02d}_{label}.mp4   one mp4 per panel (so each video stands alone)
        montage.mp4                 the panels hstacked left-to-right (same order)
        start_frame.png             the conditioning first frame (if given)
        layout.json                 panel order + file + per-panel info + metrics

    `panels`: list of dicts, each with 'label' (str) and 'pix' ([3,T,H,W] in [-1,1]); any
    other keys (e.g. 'desc','mse_to_gt') are copied into that panel's layout entry.
    Returns the layout list."""
    os.makedirs(out_dir, exist_ok=True)
    layout, pix_list = [], []
    for i, p in enumerate(panels):
        fn = f"panel_{i:02d}_{_slug(p['label'], i)}.mp4"
        write_mp4(p["pix"], os.path.join(out_dir, fn), fps=fps)
        entry = {"index": i, "label": p["label"], "file": fn}
        entry.update({k: v for k, v in p.items() if k not in ("label", "pix")})
        layout.append(entry)
        pix_list.append(p["pix"])
    write_mp4(hstack_clips(pix_list, pad=6), os.path.join(out_dir, "montage.mp4"), fps=fps)
    doc = {"montage": "montage.mp4",
           "montage_panels_left_to_right": [e["label"] for e in layout],
           "panels": layout}
    if start_frame is not None:
        save_png(start_frame, os.path.join(out_dir, "start_frame.png"))
        doc["start_frame"] = "start_frame.png"
    if meta:
        doc["metrics"] = meta
    with open(os.path.join(out_dir, "layout.json"), "w") as f:
        json.dump(doc, f, indent=2, default=str)
    return layout


def hstack_clips(clips, pad=4, pad_value=0):
    """Side-by-side montage of several [3, T, H, W] float clips in [-1,1].

    Clips may differ in T (e.g. GT has 4F pixel frames, a VAE round-trip has ~4F-3):
    they are trimmed to the shortest T. Heights are assumed equal; a `pad`-px column
    separates panels. Returns a [3, Tmin, H, sum(W)+pad*(n-1)] float tensor in [-1,1].
    """
    clips = [c if isinstance(c, torch.Tensor) else torch.as_tensor(c) for c in clips]
    clips = [c.detach().float().cpu() for c in clips]
    Tmin = min(c.shape[1] for c in clips)
    H = clips[0].shape[2]
    assert all(c.shape[2] == H for c in clips), "clips must share height for hstack"
    parts, sep = [], None
    if pad > 0:
        sep = torch.full((3, Tmin, H, pad), float(pad_value))
    for i, c in enumerate(clips):
        parts.append(c[:, :Tmin])
        if sep is not None and i < len(clips) - 1:
            parts.append(sep)
    return torch.cat(parts, dim=3)
