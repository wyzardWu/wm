"""Hold spare memory on one GPU for the incoming EYBX trainer.

Unlike hold_feed.py: leaves HEADROOM_MIB free while grabbing (a precompute
job may still be running on the card), and needs no flag file — it watches
for our own `bidirectional.train` rank on this GPU and hands memory over to
it gradually, exiting once that rank holds >40G.

Usage: python hold_eybx.py <gpu>
"""
import os
import subprocess
import sys
import time

import torch

GPU = int(sys.argv[1])
BLOCK = 2 * 1024**3 // 2          # 2G blocks (fp16 elements)
HEADROOM_MIB = int(os.environ.get("HOLD_HEADROOM_MIB", "25000"))

torch.cuda.set_device(GPU)


def free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits", "-i", str(GPU)],
                         capture_output=True, text=True).stdout.strip()
    return int(out)


def our_train_mib():
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
                           "-i", str(GPU)], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.splitlines()
    total = 0
    for line in apps:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or parts[0] != uuid:
            continue
        try:
            with open(f"/proc/{parts[1]}/cmdline") as f:
                cmd = f.read()
        except OSError:
            continue
        if ("bidirectional.train" in cmd or "bidirectional/train" in cmd
                or "ar_diffusion_gm" in cmd or "cd_gm" in cmd or "dmd_gm" in cmd or "ltx" in cmd):
            total += int(parts[2])
    return total


blocks = []
print(f"gpu{GPU}: holding with {HEADROOM_MIB}MiB headroom", flush=True)
while True:
    t = our_train_mib()
    if t > 0:
        break
    if free_mib() > HEADROOM_MIB + 3000:
        try:
            blocks.append(torch.empty(BLOCK, dtype=torch.float16, device=f"cuda:{GPU}"))
        except torch.cuda.OutOfMemoryError:
            pass
    time.sleep(4)

print(f"gpu{GPU}: trainer detected, releasing {2*len(blocks)}G gradually", flush=True)
# initial release so the trainer can load weights
for _ in range(min(20, len(blocks))):
    del blocks[0]
torch.cuda.empty_cache()
while blocks:
    if our_train_mib() > 40000:
        break
    if free_mib() < 12000:
        del blocks[:3]
        torch.cuda.empty_cache()
    time.sleep(1)
print(f"gpu{GPU}: handover complete", flush=True)
