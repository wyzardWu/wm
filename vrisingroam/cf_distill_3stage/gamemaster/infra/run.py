"""
Run / checkpoint / eval path conventions + logging for GameMaster.

Everything lives under the GameMaster root (which is on shared /nfs), so a run started
on one machine can be resumed / evaluated on another — the user grabs whichever 8-GPU
box is free. NO machine-local paths (/home, /data) are baked in here.

Layout (run_dir defaults to <GM>/models/<name>):
    <run_dir>/
        config.json               run config snapshot
        train.log                 human text log (tee of stdout)
        train_log.jsonl           one JSON row per logged train step (metrics)
        state_step{N}/            accelerate sharded resume state  (HEAVY ~50GB; rotated)
        dits/dit_step{N}.safetensors   consolidated bf16 DiT  (LIGHT ~10GB; for eval/transfer)
    <GM>/outputs/eval/<name>/step{N}/   eval artifacts (mp4s, montages, metrics.json)

Why split state vs dit: resume needs the full sharded optimizer state (huge, only the
latest few matter); eval/transfer needs only the consolidated DiT (small, keep a long
history to compare checkpoints). /nfs is near-full, so state dirs are rotated hard.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import time

GM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_ROOT = os.path.join(GM_ROOT, "models")
EVAL_ROOT = os.path.join(GM_ROOT, "outputs", "eval")


# ───────────────────────── path helpers ─────────────────────────

def resolve_run_dir(out):
    """`out` may be a bare run name ('gm_all3_v1') or a path; return an absolute dir.
    Bare names (no path separator) live under <GM>/models/."""
    if os.path.isabs(out):
        return out
    if os.sep in out:
        return os.path.join(GM_ROOT, out)
    return os.path.join(MODELS_ROOT, out)


def state_dir(run_dir, step):
    return os.path.join(run_dir, f"state_step{step}")


def dits_dir(run_dir):
    return os.path.join(run_dir, "dits")


def dit_path(run_dir, step):
    return os.path.join(dits_dir(run_dir), f"dit_step{step}.safetensors")


def eval_root_for(run_dir):
    return os.path.join(EVAL_ROOT, os.path.basename(os.path.normpath(run_dir)))


def eval_dir(run_dir, step):
    return os.path.join(eval_root_for(run_dir), f"step{step}")


def _steps_from(paths, pattern):
    out = []
    for p in paths:
        m = re.search(pattern, os.path.basename(os.path.normpath(p)))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def list_states(run_dir):
    """[(step, dir), ...] sorted ascending by step."""
    return _steps_from(glob.glob(os.path.join(run_dir, "state_step*")), r"state_step(\d+)$")


def latest_state(run_dir):
    s = list_states(run_dir)
    return s[-1] if s else None            # (step, dir) or None


def list_dits(run_dir):
    """[(step, path), ...] sorted ascending by step."""
    return _steps_from(glob.glob(os.path.join(dits_dir(run_dir), "dit_step*.safetensors")),
                       r"dit_step(\d+)\.safetensors$")


def latest_dit(run_dir):
    d = list_dits(run_dir)
    return d[-1] if d else None            # (step, path) or None


def rotate_states(run_dir, keep_last):
    """Delete oldest state_step dirs, keeping the `keep_last` highest steps. Never
    deletes the single newest. Returns list of removed dirs. (Consolidated dits in
    dits/ are NOT touched — they are the small eval history.)"""
    if keep_last is None or keep_last <= 0:
        return []
    states = list_states(run_dir)
    removed = []
    for _, d in states[:-keep_last]:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d)
    return removed


# ───────────────────────── deterministic train/val holdout ─────────────────────────
# No train/val split exists in the precompute. We derive one deterministically from the
# clip FILENAME (which encodes boss+fight+start_cell), so it is identical across machines,
# stable as clips are added/removed (the precompute is still growing), and independent of
# DataLoader worker / FSDP rank / serving order. Reuses the blake2b idiom from
# data/precomputed.py::_clip_drop. Train and eval MUST pass the same val_frac + salt.

VAL_SALT = "gm_val_v1"


def clip_is_val(path, val_frac, salt=VAL_SALT):
    """True if this clip falls in the held-out val set. Keyed on os.path.basename(path)."""
    if val_frac <= 0:
        return False
    h = hashlib.blake2b(f"{os.path.basename(path)}|{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64 < val_frac


def split_clip_files(files, split, val_frac, salt=VAL_SALT):
    """Filter a list of clip_*.pt paths to 'train' (complement) or 'val' (held out).
    split=None or val_frac=0 -> return all files unchanged (back-compat)."""
    if split is None or val_frac <= 0:
        return list(files)
    if split == "val":
        return [f for f in files if clip_is_val(f, val_frac, salt)]
    if split == "train":
        return [f for f in files if not clip_is_val(f, val_frac, salt)]
    raise ValueError(f"split must be None|'train'|'val', got {split!r}")


# ───────────────────────── balanced oversampling (game / boss) ─────────────────────────
# The v3_all clip counts are skewed both across games (VR 21.5% / HK 33.0% / Isaac 45.6%)
# and across bosses within a game (Hornet 48.6% of HK vs HiveKnight 13.5%). Uniform
# per-clip sampling inherits that skew; with the short distill schedules the rare game
# gets too little exposure. This helper rebalances by DETERMINISTIC LIST REPETITION —
# rare clips appear k(+1) times in the file list, so the unchanged shuffle=True
# RandomSampler + accelerate BatchSamplerShard path yields the target mix. Never drops
# data: factors are normalized so the most-overrepresented boss keeps factor 1.0.
# Fractional factors use the same blake2b idiom as clip_is_val, so the list is identical
# on every rank and across restarts.

BALANCE_SALT = "gm_balance_v1"


def _extra_copy(path, frac, salt=BALANCE_SALT):
    h = hashlib.blake2b(f"{os.path.basename(path)}|{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64 < frac


def oversample_clip_files(files, mode, alpha=1.0, max_repeat=8.0, log=print):
    """Return a rebalanced clip list. mode: None/'none' = unchanged; 'game' = equalize
    games (boss mix inside a game stays natural); 'game_boss' = equalize games AND bosses
    within each game. alpha in [0,1] interpolates factor = (target/natural)^alpha;
    max_repeat caps the per-boss repetition factor (log tells you when it binds)."""
    if not mode or mode == "none":
        return list(files)
    if mode not in ("game", "game_boss"):
        raise ValueError(f"balance mode must be none|game|game_boss, got {mode!r}")
    from gamemaster.data.vrising import GAME_OF_BOSS
    by_boss = {}
    for f in files:
        boss = os.path.basename(f)[len("clip_"):].split("_")[0]
        by_boss.setdefault(boss, []).append(f)
    unknown = sorted(set(by_boss) - set(GAME_OF_BOSS))
    if unknown:
        raise KeyError(f"boss prefix(es) {unknown} not in vrising.GAME_OF_BOSS — add them first")
    games = sorted({GAME_OF_BOSS[b] for b in by_boss})
    bosses_of = {g: sorted(b for b in by_boss if GAME_OF_BOSS[b] == g) for g in games}
    total = len(files)
    factor = {}
    for g in games:
        g_count = sum(len(by_boss[b]) for b in bosses_of[g])
        for b in bosses_of[g]:
            natural = len(by_boss[b]) / total
            if mode == "game":
                target = (1 / len(games)) * (len(by_boss[b]) / g_count)
            else:
                target = (1 / len(games)) * (1 / len(bosses_of[g]))
            factor[b] = (target / natural) ** alpha
    lo = min(factor.values())
    capped = []
    for b in factor:
        factor[b] /= lo                      # most-overrepresented boss stays at 1.0
        if factor[b] > max_repeat:
            capped.append(b)
            factor[b] = max_repeat
    out = []
    for b in sorted(by_boss):
        k, frac = int(factor[b]), factor[b] - int(factor[b])
        for f in by_boss[b]:
            out.extend([f] * k)
            if _extra_copy(f, frac):
                out.append(f)
    eff = {}
    for f in out:
        boss = os.path.basename(f)[len("clip_"):].split("_")[0]
        eff[GAME_OF_BOSS[boss]] = eff.get(GAME_OF_BOSS[boss], 0) + 1
    shares = " ".join(f"{g}={eff[g]/len(out):.1%}" for g in games)
    log(f"balanced sampling ON: mode={mode} alpha={alpha} epoch x{len(out)/total:.2f} "
        f"({total} -> {len(out)}); game shares: {shares}"
        + (f"; CAPPED at {max_repeat}x: {capped}" if capped else ""))
    return out


# ───────────────────────── disk guard ─────────────────────────

def disk_free_gb(path=GM_ROOT):
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("nan")


def warn_if_low_disk(path=GM_ROOT, min_gb=150):
    free = disk_free_gb(path)
    if free == free and free < min_gb:    # not NaN and low
        return (f"LOW DISK: {free:.0f}GB free on {path} (< {min_gb}GB). A 5B FSDP "
                f"state_step is ~50-60GB; reduce --keep_last_states or --save_every, "
                f"or point --out at a roomier shared volume.")
    return None


# ───────────────────────── logging ─────────────────────────

class JsonlLogger:
    """Append-only JSONL metric log. Each .log(dict) writes one flushed line so a
    watcher / another machine can tail it live."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._f = open(path, "a", buffering=1)        # line-buffered

    def log(self, row):
        row = dict(row)
        row.setdefault("wall", time.time())
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


class TextTee:
    """Print to stdout AND append to <run_dir>/train.log with a wall-clock prefix."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._f = open(path, "a", buffering=1)

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(msg, flush=True)
        self._f.write(f"[{stamp}] {msg}\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def save_config(run_dir, cfg):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


# ───────────────────────── checkpoint resolution (for rollout/eval) ─────────────────────────

def resolve_dit(ckpt):
    """Map a --ckpt argument to a consolidated DiT .safetensors path (or the literal
    'base' sentinel). Accepts:
      - 'base'                       -> 'base' (caller loads the Wan2.2 base shards)
      - a .safetensors file          -> that file
      - a run dir / run name         -> its latest dits/dit_step*.safetensors
      - a state_step / step dir      -> a sibling dit, else the newest dit in the run
    Returns (kind, path) where kind in {'base','dit'}."""
    if ckpt in (None, "base"):
        return ("base", None)
    if ckpt.endswith(".safetensors") and os.path.isfile(ckpt):
        return ("dit", ckpt)
    # a directory: try as a run dir first
    cand = ckpt if os.path.isabs(ckpt) or os.sep in ckpt else resolve_run_dir(ckpt)
    if os.path.isdir(cand):
        d = latest_dit(cand)
        if d:
            return ("dit", d[1])
        # maybe it's a state/step dir whose run is the parent
        parent = os.path.dirname(os.path.normpath(cand))
        d = latest_dit(parent)
        if d:
            return ("dit", d[1])
    raise FileNotFoundError(f"could not resolve a DiT checkpoint from --ckpt={ckpt!r}")
