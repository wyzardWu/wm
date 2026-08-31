"""Hold one GPU's memory, then hand it over to our own incoming training.

Grabs all free memory in 4G blocks. When the flag file appears, trims to leave
20G free, then releases blocks as our ar_diffusion/cd_gm rank on this GPU grows.
Exits once that rank holds >40G (handover complete).

Usage: python hold_feed.py <gpu> <flagfile>
"""
import os
import subprocess
import sys
import time

import torch

GPU = int(sys.argv[1])
FLAG = sys.argv[2]
BLOCK = 4 * 1024**3 // 2

torch.cuda.set_device(GPU)
blocks = []
while True:
    try:
        blocks.append(torch.empty(BLOCK, dtype=torch.float16, device=f"cuda:{GPU}"))
    except torch.cuda.OutOfMemoryError:
        break
print(f"gpu{GPU}: holding {4*len(blocks)}G", flush=True)


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
        if parts[0] != uuid:
            continue
        try:
            with open(f"/proc/{parts[1]}/cmdline") as f:
                cmd = f.read()
        except OSError:
            continue
        if "ar_diffusion_gm" in cmd or "cd_gm" in cmd or "dmd_gm" in cmd:
            total += int(parts[2])
    return total


while not os.path.exists(FLAG):
    try:
        blocks.append(torch.empty(BLOCK, dtype=torch.float16, device=f"cuda:{GPU}"))
    except torch.cuda.OutOfMemoryError:
        pass
    time.sleep(5)

print(f"gpu{GPU}: flag seen, trimming", flush=True)
while free_mib() < 20000 and blocks:
    del blocks[:1]
    torch.cuda.empty_cache()

while blocks:
    if our_train_mib() > 40000:
        break
    if free_mib() < 15000:
        del blocks[:3]
        torch.cuda.empty_cache()
    time.sleep(0.5)
print(f"gpu{GPU}: handover complete", flush=True)
