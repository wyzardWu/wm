#!/bin/bash
# 并行 range 下载官方 distilled 权重 + sha256 对 LFS oid(铁律)
set -u
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
DL=/data/yuzhewu/ltxwm/dl_distilled
$PY - <<'PYEOF'
import json, os, subprocess, hashlib, urllib.request
DL = "/data/yuzhewu/ltxwm/dl_distilled"
api = json.load(urllib.request.urlopen("https://huggingface.co/api/models/Lightricks/LTX-2.3/tree/main"))
targets = ["ltx-2.3-22b-distilled-1.1.safetensors",
           "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
           "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"]
info = {f["path"]: f["lfs"] for f in api if f["path"] in targets}
for name in targets:
    lfs = info[name]; size = lfs["size"]; oid = lfs["oid"]
    out = os.path.join(DL, name)
    url = f"https://huggingface.co/Lightricks/LTX-2.3/resolve/main/{name}"
    if not (os.path.exists(out) and os.path.getsize(out) == size):
        n = 16 if size > 2**33 else 4
        chunk = size // n + 1
        procs = []
        for i in range(n):
            s, e = i*chunk, min((i+1)*chunk-1, size-1)
            procs.append(subprocess.Popen(["curl","-sfL","--retry","10","--retry-all-errors",
                "-r",f"{s}-{e}","-o",f"{out}.part{i}",url]))
        codes = [p.wait() for p in procs]
        assert all(c==0 for c in codes), f"download failed {name} {codes}"
        with open(out,"wb") as w:
            for i in range(n):
                with open(f"{out}.part{i}","rb") as r:
                    while True:
                        b = r.read(1<<24)
                        if not b: break
                        w.write(b)
                os.remove(f"{out}.part{i}")
    h = hashlib.sha256()
    with open(out,"rb") as r:
        while True:
            b = r.read(1<<24)
            if not b: break
            h.update(b)
    got = h.hexdigest()
    status = "OK" if got == oid else "SHA_MISMATCH"
    print(f"{name}: {status} ({got[:16]} vs {oid[:16]})", flush=True)
    assert got == oid, name
print("ALL_DOWNLOADS_VERIFIED")
PYEOF
