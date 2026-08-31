"""EYBX 切场景实时交互 demo:浏览器按键 → 逐潜帧 AR 生成 → JPEG 推流。

复用 s1_ar_rollout.py 的全部装载/推理路径,唯一区别:每个潜帧的 ctx 来自
玩家当前按键状态而不是预排 parquet。单会话(一次一个玩家)。

用法: CUDA_VISIBLE_DEVICES=6 python play_server.py [--port 8899] [--steps 2]
玩家: ssh -L 8899:localhost:8899 yuzhewu@157.10.162.252 → http://localhost:8899
键位: WASD 移动 | 1=地牢 2=岩窟 3=林道 0=null | R 重置 | 下拉选种子图
"""
import argparse
import asyncio
import io
import json
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/home/yuzhewu/vrisingroam")
sys.path.insert(0, str(ROOT / "ReactiveGWM"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cf_distill_3stage"))

from PIL import Image
from aiohttp import web, WSMsgType
from safetensors.torch import load_file

from ReactiveGWM_Code.inference.models import WanVideoVAE38
from ReactiveGWM_Code.inference.utils import preprocess_image, to_pil_video
from gamemaster.models.gamemaster_dit import GameMasterDiT, ti2v_5b_config
from gamemaster.flow_match import FlowMatchScheduler

EMA = "/data/yuzhewu/vrisingroam/distill/runs/eybx_cd_v1/ema/ema_step2000.safetensors"
TBL = "/data/yuzhewu/eybxroam/tables/eybx_action_table.pt"
STBL = "/data/yuzhewu/eybxroam/tables/eybx_scene_table.pt"
SEEDS = Path("/data/yuzhewu/eybxroam/probe_seeds")
BASE = "/nfs/zeqingwang/models/base_model"
DEV, DT = "cuda", torch.bfloat16
MAX_K_ABS = 25            # 训练协议 101px = 26 latent
DECODE_WIN = 2


class Engine:
    def __init__(self, steps, kv_window):
        self.steps, self.kv_window = steps, kv_window
        self.lock = threading.Lock()
        vae = WanVideoVAE38()
        vs = torch.load(Path(BASE) / "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
                        map_location="cpu", weights_only=True)
        vae.load_state_dict({f"model.{k}": v for k, v in vs.items()}, strict=False)
        self.vae = vae.to(DT).eval().requires_grad_(False).to(DEV)
        model = GameMasterDiT(**ti2v_5b_config(), causal=True, zero_init_head=False)
        sd = load_file(EMA)
        info = model.load_state_dict(sd, strict=False)
        assert not info.missing_keys and not info.unexpected_keys
        self.model = model.to(DT).eval().requires_grad_(False).to(DEV)
        tb = torch.load(TBL, map_location="cpu", weights_only=False)
        self.table = (tb["table"] if isinstance(tb, dict) else tb).to(DEV, DT)
        sb = torch.load(STBL, map_location="cpu", weights_only=False)
        self.stable = sb["table"].to(DEV, DT)
        self.sched = FlowMatchScheduler(shift=5.0)
        self.sched.set_timesteps(steps, device=DEV)
        self.gen = torch.Generator("cpu").manual_seed(7)
        self.k = 0
        self.cache = None
        self.latents = []
        print("[play] engine ready", flush=True)

    def ctx_row(self, keys, scene):
        aid = (keys.get("W", 0) * 1 + keys.get("A", 0) * 2 + keys.get("S", 0) * 4
               + keys.get("D", 0) * 8 + keys.get("M", 0) * 16)
        row = torch.cat([self.table[aid], self.stable[scene]], dim=0)  # [32,4096]
        return row[None, None]                                         # [1,1,32,4096]

    @torch.no_grad()
    def reset(self, seed_name, keys, scene):
        img = preprocess_image(Image.open(SEEDS / seed_name), 480, 832,
                               device=DEV, dtype=DT)
        lat0 = self.vae.encode([img], device=DEV).to(DEV, DT)
        if self.kv_window > 0:
            self.cache = self.model.init_kv_cache(kv_window=self.kv_window,
                                                  rope_cap=16, sink_size=1)
        else:
            self.cache = self.model.init_kv_cache()
        zero_t = torch.zeros(1, device=DEV)
        self.model(lat0, zero_t, self.ctx_row(keys, scene), frame_index=0,
                   kv_cache=self.cache, commit=True)
        self.latents = [lat0]
        self.k = 1
        self.seg = 1
        self.last_pil = None

    @torch.no_grad()
    def rollover(self, keys, scene):
        """无缝续段:最后一帧生成画面 -> 新种子,重开 KV 窗口。"""
        lat0 = self.vae.encode([preprocess_image(self.last_pil, 480, 832,
                                                 device=DEV, dtype=DT)],
                               device=DEV).to(DEV, DT)
        self.cache = self.model.init_kv_cache()
        self.model(lat0, torch.zeros(1, device=DEV), self.ctx_row(keys, scene),
                   frame_index=0, kv_cache=self.cache, commit=True)
        self.latents = [lat0]
        self.k = 1
        self.seg += 1
        print(f"[play] rollover -> segment {self.seg}", flush=True)

    @torch.no_grad()
    def step(self, keys, scene):
        """一潜帧:2 步去噪+commit,窗口解码,返回最后 4 帧 PIL。"""
        if self.kv_window == 0 and self.k > MAX_K_ABS:
            self.rollover(keys, scene)
        self._t0 = time.time()
        ctx = self.ctx_row(keys, scene)
        h, w = self.latents[0].shape[3], self.latents[0].shape[4]
        x = torch.randn((1, 48, 1, h, w), generator=self.gen,
                        dtype=torch.float32).to(DEV, DT)
        for i in range(self.steps):
            t = self.sched.infer_timesteps[i].reshape(1).to(DEV)
            v = self.model(x, t, ctx, frame_index=self.k, kv_cache=self.cache,
                           commit=False)
            x = self.sched.step(v, x, i)
        self.model(x, torch.zeros(1, device=DEV), ctx, frame_index=self.k,
                   kv_cache=self.cache, commit=True)
        t_gen = time.time()
        self.latents.append(x)
        self.k += 1
        win = torch.cat(self.latents[-DECODE_WIN:], dim=2)
        video = self.vae.decode(win, device=DEV)
        out = to_pil_video(video)[-4:]
        self.last_pil = out[-1]
        if self.k % 8 == 0:
            print(f"[play] k={self.k} gen={t_gen - self._t0:.2f}s "
                  f"decode={time.time() - t_gen:.2f}s", flush=True)
        return out


INDEX_HTML = """<!doctype html><meta charset=utf-8><title>EYBX play</title>
<body style="background:#0d1117;color:#eee;font-family:monospace;text-align:center">
<h3>EYBX 切场景实时交互 <span id=st style="color:#2dd4bf"></span></h3>
<canvas id=cv width=832 height=480 style="border:1px solid #333"></canvas>
<div style="margin:8px">WASD 移动 · 1 地牢 / 3 林道 / 0 null · R 重置 ·
种子 <select id=seed>SEED_OPTIONS</select>
<span id=info style="color:#969ea8"></span></div>
<script>
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const ws = new WebSocket('ws://' + location.host + '/ws');
ws.binaryType = 'arraybuffer';
let keys = {W:0,A:0,S:0,D:0,M:0}, scene = 0, n = 0, t0 = Date.now();
let capped = false;
const SCENES = ['1 地牢', '2 岩窟', '3 林道', '0 null'];
const q = [];
function hud() {
  ctx.fillStyle = 'rgba(0,0,0,.6)'; ctx.fillRect(0, 0, 200, 30);
  ctx.fillStyle = '#2dd4bf'; ctx.font = 'bold 18px monospace';
  ctx.fillText('场景 ' + SCENES[scene], 10, 21);
  if (capped) {
    ctx.fillStyle = 'rgba(0,0,0,.75)'; ctx.fillRect(0, 190, 832, 100);
    ctx.fillStyle = '#ffd23f'; ctx.font = 'bold 34px monospace';
    ctx.fillText('本段到头了 — 按 R 重新开始', 160, 250);
  }
}
ws.onmessage = e => {
  if (typeof e.data === 'string') {
    document.getElementById('st').innerText = e.data;
    if (e.data.includes('上限')) { capped = true; hud(); }
    return; }
  q.push(e.data); n++;
  while (q.length > 4) q.shift();   // 丢旧帧防延迟累积
  document.getElementById('info').innerText =
    ' 帧 ' + n + ' | ' + (1000 * n / (Date.now() - t0)).toFixed(1) + ' fps';
};
setInterval(() => {
  if (!q.length) return;
  const blob = new Blob([q.shift()], {type: 'image/jpeg'});
  createImageBitmap(blob).then(b => { ctx.drawImage(b, 0, 0, 832, 480); hud(); });
}, 55);
function send() { if (ws.readyState === 1) ws.send(JSON.stringify({type:'keys', keys, scene})); }
const km = {w:'W', a:'A', s:'S', d:'D'};
onkeydown = e => {
  const k = e.key.toLowerCase();
  if (km[k]) { keys[km[k]] = 1; send(); }
  if ('013'.includes(k)) { scene = {'1':0,'3':2,'0':3}[k]; send(); }
  if (k === 'r') { n = 0; t0 = Date.now(); capped = false;
    const sd = document.getElementById('seed').value;
    scene = sd.includes('infiniteDungeon') ? 0 : sd.includes('mountainPass') ? 1
          : sd.includes('hills') ? 2 : 3;
    send();
    ws.send(JSON.stringify({type:'reset', seed:sd})); }
};
onkeyup = e => { const k = e.key.toLowerCase(); if (km[k]) { keys[km[k]] = 0; send(); } };
ws.onopen = () => {
  const sd = document.getElementById('seed').value;
  scene = sd.includes('infiniteDungeon') ? 0 : sd.includes('mountainPass') ? 1
        : sd.includes('hills') ? 2 : 3;
  ws.send(JSON.stringify({type:'reset', seed:sd}));
};
</script>"""


async def index(request):
    opts = "".join(f"<option>{p.name}</option>" for p in sorted(SEEDS.glob("*.png")))
    return web.Response(text=INDEX_HTML.replace("SEED_OPTIONS", opts),
                        content_type="text/html")


async def ws_handler(request):
    eng: Engine = request.app["engine"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    if not eng.lock.acquire(blocking=False):
        await ws.send_str("已有玩家在线,稍后再试")
        await ws.close()
        return ws
    state = {"keys": {}, "scene": 0, "running": False}
    loop = asyncio.get_event_loop()

    async def gen_loop():
        try:
            while state["running"] and not ws.closed:
                t0 = time.time()
                frames = await loop.run_in_executor(
                    None, eng.step, dict(state["keys"]), state["scene"])
                if frames is None:
                    await ws.send_str(f"到达协议长度上限(k={eng.k - 1}),按 R 重置")
                    state["running"] = False
                    break
                for im in frames:
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=82)
                    await ws.send_bytes(buf.getvalue())
                await ws.send_str(f"k={eng.k - 1} scene={state['scene']} "
                                  f"{time.time() - t0:.2f}s/潜帧")
        except Exception as e:
            print(f"[play] gen_loop ended: {type(e).__name__}: {e}", flush=True)
        finally:
            state["running"] = False

    task = None
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            d = json.loads(msg.data)
            if d["type"] == "keys":
                state["keys"] = d["keys"]
                state["scene"] = int(d["scene"])
            elif d["type"] == "reset":
                state["running"] = False
                if task:
                    await task
                await loop.run_in_executor(
                    None, eng.reset, d["seed"], dict(state["keys"]), state["scene"])
                state["running"] = True
                task = asyncio.ensure_future(gen_loop())
    finally:
        state["running"] = False
        if task:
            try:
                await task
            except Exception as e:
                print(f"[play] task join: {type(e).__name__}", flush=True)
        eng.lock.release()
        print("[play] session released", flush=True)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--kv_window", type=int, default=0,
                    help=">0 = sliding KV 无限漫游(CD 学生兼容性未验)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile DiT(首潜帧编译预热 ~2min,之后 1.5-2x)")
    args = ap.parse_args()
    app = web.Application()
    eng = Engine(args.steps, args.kv_window)
    if args.compile:
        eng.model = torch.compile(eng.model, mode="max-autotune-no-cudagraphs")
        print("[play] torch.compile armed (first latent will be slow)", flush=True)
    app["engine"] = eng
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
