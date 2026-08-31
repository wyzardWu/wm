"""GameMaster Stage-2 (Option B) — Causal Consistency Distillation (CD) init.

SECOND distillation route (scheme 2), adapted from thu-ml Causal-Forcing++ (CF++,
arXiv 2605.15141; vendored read-only at vendor/causal_forcing/). It REPLACES the
ODE-init (distill/ode_init_gm.py) Stage-2a: a few-step causal student is produced
WITHOUT generating offline ODE paired data — using only GT latents + the teacher's
on-the-fly single Euler step. Output feeds Stage-2b DMD (distill/dmd_gm.py) exactly
like the ODE-init did.

CD loss (faithful to vendor/causal_forcing/model/naive_consistency.py:85, realized in
our NATIVE conventions = ode_init_gm/dmd_gm: channels-first, C9 per-frame text,
frame0-clean, full-clip causal forward — NO rollout, NO critic):
  1. take a GT clean clip; discrete consistency axis of N steps (FlowMatch shift=5).
  2. pick idx; sigma_t = sigmas[idx] (higher noise), sigma_tn = sigmas[idx+1] (lower).
  3. noise the clip at sigma_t -> latent_t (frame0 forced clean).
  4. FROZEN teacher (CFG) predicts velocity; ONE Euler step t->t_next:
        latent_tn = latent_t + (sigma_tn - sigma_t) * v_pred   (frame0 kept clean)
  5. student  x0 estimate at t      :  cm_t  = latent_t  - sigma_t  * student(latent_t , t )
  6. student-EMA x0 estimate at t_nx:  cm_tn = latent_tn - sigma_tn * ema(    latent_tn, t_nx)   [no grad]
  7. loss = MSE(cm_t[:,:,1:], cm_tn[:,:,1:].detach())   — local self-consistency along the
     teacher's causal ODE flow. frame0 (clean i2v cond) excluded.

Two configs (same code, --teacher_causal selects):
  P2 (pragmatic, priority 1): teacher = our BIDIR teacher gm_all3_v1 (--teacher_causal 0),
      student init = gm_ode_v2 ODE-init. cfg 1.0 (our teacher has NO text dropout, see
      dmd_gm --real_guidance_scale note). Reuses existing assets; less faithful (bidir
      teacher trajectory != causal student) but cheap to validate.
  P1 (faithful, priority 2): teacher = a CAUSAL multi-step AR-diffusion GameMaster
      (--teacher_causal 1) produced by distill/ar_diffusion_gm.py; student init = that
      same causal ckpt. Matches CF++ theory (师生同结构/同轨迹).

Generator-EMA (CF deploys the EMA weights; infer with the ema_step*.safetensors): under
FSDP with fsdp_use_orig_params=true, student.parameters() and ema.parameters() are the
SAME original Parameter objects in identical order, identically sharded -> per-shard EMA
via a plain zip() is exact.

CPU smoke (no GPU/FSDP):  python distill/cd_gm.py --smoke

★ FSDP launch (3 full 5B models = student+ema+teacher): use the 3-MODEL config
scripts/accelerate_fsdp_dmd.yaml (sync_module_states=false + cpu_ram_efficient_loading=false —
each rank load_dit_ckpt's the SAME ckpt deterministically, so rank0 must NOT broadcast; the
default scripts/accelerate_fsdp.yaml has sync_module_states=true → rank0 materializes all 3
5B models → ~75GB rank0 prepare OOM on <100GB cards). The launcher distill/run_cd_gm.sh already
defaults FSDP_CFG to accelerate_fsdp_dmd.yaml. Manual:
    accelerate launch --config_file scripts/accelerate_fsdp_dmd.yaml --num_processes 4 \\
        distill/cd_gm.py --teacher_ckpt ... --student_ckpt ... --out ...
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

GM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GM)
from gamemaster.models.gamemaster_dit import GameMasterDiT, ti2v_5b_config, tiny_config
from gamemaster.flow_match import FlowMatchScheduler


# ───────────────────────── consistency axis + noising ─────────────────────────

def cd_axis(sched, n, dev):
    """Discrete consistency axis: N sigmas/timesteps (descending, shift-warped). Returns
    (sigmas[N], timesteps[N]) on dev. sigmas[0]~1 (high noise) -> sigmas[N-1]~0.1 (low)."""
    sched.set_timesteps(num_inference_steps=n, denoising_strength=1.0, device=dev)
    return sched.infer_sigmas.to(dev, torch.float32), sched.infer_timesteps.to(dev, torch.float32)


def _per_frame_t(t_val, B, Fl, dev):
    """[B,Fl] per-frame timestep, frame0 clean (t=0), frames 1..F-1 at t_val."""
    ts = torch.full((B, Fl), float(t_val), device=dev, dtype=torch.float32)
    ts[:, 0] = 0.0
    return ts


def noise_clip(clip, sigma, noise):
    """latent_t = (1-sigma)*clip + sigma*noise, frame0 forced clean. sigma: scalar tensor."""
    latent = (1 - sigma) * clip + sigma * noise
    f0 = (torch.arange(clip.shape[2], device=clip.device) == 0).view(1, 1, -1, 1, 1)
    return torch.where(f0, clip, latent)


def _keep_frame0(x, clip):
    f0 = (torch.arange(clip.shape[2], device=clip.device) == 0).view(1, 1, -1, 1, 1)
    return torch.where(f0, clip, x)


# ───────────────────────── CD generator loss ─────────────────────────

def cd_generator_loss(student, ema, teacher, clip, context, uncond, sigmas, tsteps,
                      idx, real_guidance_scale, gen=None, teacher_forcing=False):
    """Consistency-distillation loss (see module docstring). Returns (loss, log).

    teacher_forcing (CF/CF++ parity, naive_consistency.py:110-136): pass the UNMODIFIED clean GT
    clip as a clean_x context channel into ALL forwards (teacher cond/uncond, student, EMA). The
    clean prefix is invariant across the consistency step — only the noisy `x` and timestep change
    (the EMA takes x=latent_tn but clean_x stays the original `clip`). clean_x=None ⇒ the model's
    Diffusion-Forcing path (current behavior, byte-identical). No permute: we are channels-first
    end-to-end (unlike CF's wan_wrapper.py:259)."""
    B, C, Fl = clip.shape[:3]
    dev = clip.device
    cx = clip if teacher_forcing else None                       # CF clean_x = the GT clean clip
    sig_t, sig_tn = sigmas[idx], sigmas[idx + 1]                 # scalar tensors (sig_tn < sig_t)
    t_val, tn_val = float(tsteps[idx]), float(tsteps[idx + 1])
    s_t = sig_t.view(1, 1, 1, 1, 1)
    s_tn = sig_tn.view(1, 1, 1, 1, 1)

    noise = torch.randn(clip.shape, device=dev, dtype=clip.dtype, generator=gen)
    latent_t = noise_clip(clip, s_t, noise)
    ts_t = _per_frame_t(t_val, B, Fl, dev)
    ts_tn = _per_frame_t(tn_val, B, Fl, dev)

    # teacher single Euler step t -> t_next (CFG); cfg==1 => only conditional forward needed
    with torch.no_grad():
        v_c = teacher(latent_t, ts_t, context, clean_x=cx)
        if abs(real_guidance_scale - 1.0) < 1e-6:
            v_pred = v_c
        else:
            v_u = teacher(latent_t, ts_t, uncond, clean_x=cx)
            v_pred = v_u + real_guidance_scale * (v_c - v_u)
        latent_tn = latent_t + (s_tn - s_t) * v_pred             # Euler to lower noise
        latent_tn = _keep_frame0(latent_tn, clip)

    # student consistency x0 at t (carries grad)
    v_s = student(latent_t, ts_t, context, clean_x=cx)
    cm_t = latent_t - s_t * v_s
    # EMA consistency x0 at t_next (no grad); CF: x=latent_tn but clean_x stays the original clip
    with torch.no_grad():
        v_e = ema(latent_tn, ts_tn, context, clean_x=cx)
        cm_tn = latent_tn - s_tn * v_e

    loss = F.mse_loss(cm_t[:, :, 1:].float(), cm_tn[:, :, 1:].detach().float())
    return loss, {"cd_loss": loss.item(), "sigma_t": float(sig_t), "sigma_tn": float(sig_tn)}


# ───────────────────────── generator EMA (FSDP use_orig_params) ─────────────────────────

@torch.no_grad()
def ema_update(ema, student, decay):
    """Per-shard EMA. With fsdp_use_orig_params=true both models expose the SAME original
    Parameters in identical order, sharded identically -> zip is exact. decay<=0 => hard copy."""
    for pe, p in zip(ema.parameters(), student.parameters()):
        if decay <= 0.0:
            pe.data.copy_(p.data)
        else:
            pe.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ───────────────────────── CPU smoke ─────────────────────────

def smoke():
    torch.manual_seed(0)
    cfg = tiny_config()
    dev = "cpu"
    teacher = GameMasterDiT(**cfg, causal=False, zero_init_head=False).eval().requires_grad_(False).to(dev)
    student = GameMasterDiT(**cfg, causal=True, zero_init_head=False).to(dev)
    ema = GameMasterDiT(**cfg, causal=True, zero_init_head=False).eval().requires_grad_(False).to(dev)
    student.load_state_dict(teacher.state_dict())
    ema.load_state_dict(teacher.state_dict())
    sched = FlowMatchScheduler(shift=5.0).to(dev)
    N = 8
    sigmas, tsteps = cd_axis(sched, N, torch.device(dev))
    print(f"consistency axis N={N}: sigma[0]={sigmas[0]:.3f} .. sigma[-1]={sigmas[-1]:.3f}; "
          f"t[0]={tsteps[0]:.0f} .. t[-1]={tsteps[-1]:.0f}")

    B, C, Fl, Hp, Wp, L = 1, 48, 5, 8, 8, 4
    clip = torch.randn(B, C, Fl, Hp, Wp)
    context = torch.randn(B, Fl, L, 4096)
    uncond = torch.randn(B, Fl, L, 4096)
    idx = N // 2

    print("── CD generator loss + backward ──")
    gl, glog = cd_generator_loss(student, ema, teacher, clip, context, uncond,
                                 sigmas, tsteps, idx, real_guidance_scale=1.0)
    gl.backward()
    gp = [p for p in student.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"cd loss={gl.item():.5f}  {glog}; student params with grad: {len(gp)} (must be >0)")
    assert len(gp) > 0, "CD loss did not reach student params!"
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in teacher.parameters()), "teacher got grad!"
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in ema.parameters()), "ema got grad!"

    print("── EMA update (hard copy then decay) ──")
    before = next(ema.parameters()).data.clone()
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p) * 0.1)                    # perturb student
    ema_update(ema, student, decay=0.0)                          # hard copy
    after_copy = next(ema.parameters()).data.clone()
    assert not torch.allclose(before, after_copy), "EMA hard-copy did nothing"
    ema_update(ema, student, decay=0.99)                         # decay
    print("EMA copy + decay OK")

    # cfg!=1 path (uses uncond)
    gl2, _ = cd_generator_loss(student, ema, teacher, clip, context, uncond,
                               sigmas, tsteps, idx, real_guidance_scale=2.0)
    assert torch.isfinite(gl2), "cfg!=1 path produced non-finite loss"
    print(f"cfg=2.0 path loss={gl2.item():.5f} (finite ok)")

    # teacher_forcing path (CF parity): clean_x=clip threaded into ALL forwards. Needs a CAUSAL teacher.
    print("── teacher_forcing=True (clean_x threaded; causal teacher) ──")
    teacher_c = GameMasterDiT(**cfg, causal=True, zero_init_head=False).eval().requires_grad_(False).to(dev)
    teacher_c.load_state_dict(student.state_dict())
    for p in student.parameters():
        if p.grad is not None:
            p.grad = None
    glt, gltlog = cd_generator_loss(student, ema, teacher_c, clip, context, uncond,
                                    sigmas, tsteps, idx, real_guidance_scale=1.0, teacher_forcing=True)
    glt.backward()
    gpt = [p for p in student.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert torch.isfinite(glt) and len(gpt) > 0, "TF CD loss did not reach student / non-finite"
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in teacher_c.parameters()), "TF teacher got grad!"
    print(f"TF cd loss={glt.item():.5f}  {gltlog}; student params with grad: {len(gpt)} (>0 ok)")

    print("\n✅ CD core OK: consistency loss reaches the student (teacher+ema frozen); "
          "EMA copy/decay work; CFG on/off both finite; teacher_forcing(clean_x) path finite + "
          "grad reaches student, teacher frozen. frame0-clean + C9 per-frame text honored.")


# ───────────────────────── trainer (Stage 2, Option B) — 3-model FSDP ─────────────────────────

def load_dit_ckpt(model, ckpt_path):
    import safetensors.torch as st
    sd = st.load_file(ckpt_path)
    info = model.load_state_dict(sd, strict=False, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"ckpt mismatch {ckpt_path}: missing={info.missing_keys[:6]} unexpected={info.unexpected_keys[:6]}")
    return model


def main():
    GMroot = GM
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(GMroot, "data/precomputed/all3_F21"))
    ap.add_argument("--v4_prompts", default=None,
                    help="v4.1 orthogonal v4_prompts.pt (per-clip cleaned player_nl/boss_nl + keep). "
                         "Set with --v4_text_table to train on the v4 clean orthogonal vocabulary.")
    ap.add_argument("--v4_text_table", default=None,
                    help="v4 UMT5 text_table.pt (full-sentence emb) — REQUIRED with --v4_prompts.")
    ap.add_argument("--move_sidecar", default=None,
                    help="fine-prompt move sidecar (move_sidecar.pt); unset = coarse prompts")
    ap.add_argument("--coarse_frac", type=float, default=0.0)
    ap.add_argument("--balance", default="none", choices=["none", "game", "game_boss"],
                    help="rebalance train clip sampling across games/bosses (deterministic oversampling)")
    ap.add_argument("--balance_alpha", type=float, default=1.0)
    ap.add_argument("--balance_max_repeat", type=float, default=8.0)
    ap.add_argument("--teacher_ckpt", required=True,
                    help="CD teacher. P2: gm_all3_v1 bidir dit_step{N}. P1: causal AR-diffusion dit_step{N}.")
    ap.add_argument("--student_ckpt", required=True,
                    help="generator+EMA init. P2: gm_ode_v2 ODE-init. P1: the causal AR-diffusion ckpt.")
    ap.add_argument("--teacher_causal", type=int, default=0,
                    help="0=P2 (bidir teacher gm_all3_v1). 1=P1 (causal AR-diffusion teacher).")
    ap.add_argument("--teacher_forcing", action=argparse.BooleanOptionalAction, default=False,
                    help="CF/CF++ parity: thread the clean GT clip as a clean_x context channel into ALL "
                         "(teacher/student/ema) forwards (naive_consistency.py:110-136). OFF (default) = "
                         "current Diffusion-Forcing CD (clean_x=None, unchanged). Turn ON ONLY when the "
                         "teacher+student were trained teacher-forced (ar_diffusion_gm --teacher_forcing).")
    ap.add_argument("--windowed_train", type=int, default=0,
                    help="0=OFF (full-causal mask, unchanged). >0 = trailing-window training self-attn mask "
                         "(CF long_video local_attn_size) on student+ema+teacher; for long-video CD. "
                         "RoPE stays absolute. Use AFTER TF lands.")
    ap.add_argument("--windowed_sink", type=int, default=0,
                    help="[windowed_train only] always-visible early (sink) frames (CF sink_size; 0 in CF config).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--discrete_cd_N", type=int, default=48, help="consistency-axis steps (CF default 48)")
    ap.add_argument("--real_guidance_scale", type=float, default=1.0,
                    help="teacher CFG. CF uses 3.0, but our teacher has NO text dropout "
                         "(scripts/train.py) => x0_cond≈x0_uncond, so 3.0 amplifies near-zero noise "
                         "residual. 1.0 (cond-only, mild) — same lesson as dmd_gm.")
    ap.add_argument("--ema_decay", type=float, default=0.99, help="generator EMA decay (CF ema_weight 0.99)")
    ap.add_argument("--ema_start", type=int, default=200, help="hard-copy EMA<-student until this step, then decay")
    ap.add_argument("--lr", type=float, default=2e-6, help="generator lr (CF causal_cd lr 2e-6)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=5000, help="CF recommends >=3K (more is better)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--opt8bit", type=int, default=1, help="1=bitsandbytes AdamW8bit (fit 3x5B), 0=fp32 AdamW")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--keep_last_states", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    import glob
    import time
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    sys.path.insert(0, GMroot)
    from gamemaster.data.precomputed import PrecomputedRenderDataset, collate_precomputed
    from gamemaster.infra import run as runlib

    accel = Accelerator(gradient_accumulation_steps=args.grad_accum, step_scheduler_with_optimizer=False)
    set_seed(args.seed)
    is_main = accel.is_main_process
    run_dir = runlib.resolve_run_dir(args.out)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        runlib.save_config(run_dir, {**vars(args), "world_size": accel.num_processes, "stage": "cd_gm"})
        _tee = runlib.TextTee(os.path.join(run_dir, "train.log"))
        _jsonl = runlib.JsonlLogger(os.path.join(run_dir, "train_log.jsonl"))
        log = _tee
    else:
        _jsonl = None
        def log(*a):
            pass
    log(f"=== GM CD-init (Stage 2 Option B, scheme 2) | run={run_dir} world={accel.num_processes} "
        f"teacher_causal={args.teacher_causal} teacher_forcing={args.teacher_forcing} "
        f"windowed_train={args.windowed_train} windowed_sink={args.windowed_sink} "
        f"N={args.discrete_cd_N} cfg={args.real_guidance_scale} "
        f"ema={args.ema_decay} lr={args.lr} max_steps={args.max_steps} ===")

    cfg = tiny_config() if args.dry_run else ti2v_5b_config()
    # 3 models: student(causal,trainable,FSDP) + ema(causal,frozen,FSDP) + teacher(frozen,FSDP)
    # windowed-training params only take effect on causal=True models (no-op on the bidir teacher).
    win = dict(train_local_attn_size=args.windowed_train, train_sink_size=args.windowed_sink)
    student = GameMasterDiT(**cfg, causal=True, zero_init_head=False, **win)
    ema = GameMasterDiT(**cfg, causal=True, zero_init_head=False, **win)
    teacher = GameMasterDiT(**cfg, causal=bool(args.teacher_causal), zero_init_head=False, **win)
    student.use_gradient_checkpointing = True

    # resume: load latest saved student + ema (into raw modules before prepare) + step
    resume_step = 0
    if args.resume and not args.dry_run:
        import json
        pj = os.path.join(run_dir, "progress.json")
        if os.path.exists(pj):
            resume_step = int(json.load(open(pj))["step"])
            args.student_ckpt = runlib.dit_path(run_dir, resume_step)
            ema_resume = os.path.join(run_dir, "ema", f"ema_step{resume_step}.safetensors")
            log(f"RESUME step {resume_step}: student<-{os.path.basename(args.student_ckpt)} ema<-{os.path.basename(ema_resume)}")
        else:
            log(f"--resume set but no progress.json in {run_dir}; starting fresh"); args.resume = None

    if not args.dry_run:
        if resume_step:
            load_dit_ckpt(student, args.student_ckpt)
            load_dit_ckpt(ema, ema_resume)
        else:
            log("loading: student<-init, ema<-init (same), teacher<-teacher_ckpt ...")
            load_dit_ckpt(student, args.student_ckpt)
            load_dit_ckpt(ema, args.student_ckpt)
        load_dit_ckpt(teacher, args.teacher_ckpt)
        all_files = sorted(glob.glob(os.path.join(args.data, "clip_*.pt")))
        train_files = runlib.split_clip_files(all_files, "train", args.val_frac)
        train_files = runlib.oversample_clip_files(train_files, args.balance, args.balance_alpha,
                                                   args.balance_max_repeat, log=log)
        if args.v4_prompts:
            from gamemaster.data.v4prompt import V4PromptDataset
            assert args.v4_text_table, "--v4_prompts requires --v4_text_table"
            ds = V4PromptDataset(args.data, args.v4_prompts, args.v4_text_table,
                                 boss_dropout=0.0, seed=args.seed + accel.process_index,
                                 files=train_files, validate_coverage=False)
            log(f"v4 CLEAN orthogonal prompts ON: {args.v4_prompts} table={args.v4_text_table} "
                f"kept={len(ds)}")
        else:
            ds = PrecomputedRenderDataset(args.data, boss_dropout=0.0, seed=args.seed + accel.process_index,
                                          files=train_files, validate_coverage=False,
                                          move_sidecar=args.move_sidecar, coarse_frac=args.coarse_frac)
            if args.move_sidecar:
                log(f"fine prompts ON: sidecar={args.move_sidecar} coarse_frac={args.coarse_frac}")
        neg = torch.load(os.path.join(GMroot, "common/neg_emb.pt"), map_location="cpu", weights_only=False)["neg_emb"]
        log(f"clips {len(all_files)} -> {len(train_files)} train; dataset {len(ds)}; neg_emb {tuple(neg.shape)}")
    else:
        class _Syn(torch.utils.data.Dataset):
            def __len__(self): return 64
            def set_epoch(self, e): self.epoch = e
            epoch = 0
            def __getitem__(self, i):
                g = torch.Generator().manual_seed(i)
                return {"latent": torch.randn(48, 6, 8, 8, generator=g, dtype=torch.bfloat16),
                        "context": torch.randn(6, 8, 4096, generator=g, dtype=torch.bfloat16),
                        "boss_dropped": False, "fight": 0, "start_cell": i}
        ds = _Syn()
        neg = torch.randn(8, 4096, dtype=torch.bfloat16)
        if args.max_steps > 30:
            args.max_steps, args.save_every, args.warmup, args.log_every, args.ema_start = 12, 6, 2, 2, 3
            args.discrete_cd_N = 8
        log("DRY RUN: tiny model, synthetic data")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate_precomputed, drop_last=True, pin_memory=True)

    def _adamw(params, lr):
        if args.opt8bit and not args.dry_run:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.0, betas=(0.0, 0.999))
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.0, 0.999))
    opt_g = _adamw(student.parameters(), args.lr)

    def lr_lambda(s):
        return (s + 1) / args.warmup if s < args.warmup else 1.0
    slr_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)

    teacher.requires_grad_(False).eval()
    ema.requires_grad_(False).eval()
    student, ema, teacher, opt_g, dl, slr_g = accel.prepare(student, ema, teacher, opt_g, dl, slr_g)
    neg = neg.to(accel.device, dtype=torch.bfloat16)
    sched = FlowMatchScheduler(shift=5.0).to(accel.device)
    sigmas, tsteps = cd_axis(sched, args.discrete_cd_N, accel.device)
    ac = lambda: torch.autocast(device_type=accel.device.type, dtype=torch.bfloat16)

    step = resume_step
    _ls, _lt = step, time.time()

    def _clip(model, max_norm):
        if hasattr(model, "clip_grad_norm_"):
            return model.clip_grad_norm_(max_norm)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # rng for idx sampling (same idx across ranks each step -> identical consistency level)
    idx_rng = torch.Generator(device="cpu").manual_seed(1234 + args.seed)
    if resume_step:
        # one idx is drawn per MICRO-batch (inside accel.accumulate), so an optimizer step
        # consumes grad_accum draws — fast-forward by resume_step*grad_accum to re-sync the stream.
        for _ in range(resume_step * args.grad_accum):
            torch.randint(0, args.discrete_cd_N - 1, (1,), generator=idx_rng)

    student.train()
    data_iter = iter(dl)
    last_gl = torch.tensor(float("nan"), device=accel.device)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            ds.set_epoch(getattr(ds, "epoch", 0) + 1); data_iter = iter(dl); batch = next(data_iter)
        with accel.accumulate(student):
            clip = batch["latent"].to(accel.device)                # [B,48,F,Hl,Wl]
            context = batch["context"].to(accel.device)            # [B,F,512,4096]
            B, _, Fl = clip.shape[:3]
            uncond = neg.view(1, 1, *neg.shape).expand(B, Fl, *neg.shape)
            idx = int(torch.randint(0, args.discrete_cd_N - 1, (1,), generator=idx_rng).item())
            with ac():
                gl, glog = cd_generator_loss(student, ema, teacher, clip, context, uncond,
                                             sigmas, tsteps, idx, args.real_guidance_scale,
                                             teacher_forcing=args.teacher_forcing)
            accel.backward(gl)
            gng = None
            if accel.sync_gradients:
                gng = _clip(student, 10.0)                          # CF max_grad_norm_generator=10.0
            opt_g.step(); opt_g.zero_grad()
        if accel.sync_gradients:
            slr_g.step()
            ema_update(ema, student, decay=(0.0 if step < args.ema_start else args.ema_decay))
            step += 1
            last_gl = gl.detach()
            if step % args.log_every == 0:
                avg = accel.gather(last_gl).nanmean().item()
                now = time.time(); sps = (step - _ls) / max(now - _lt, 1e-6); _ls, _lt = step, now
                log(f"step {step:6d}/{args.max_steps}  cd {avg:.5f}  sig_t {glog['sigma_t']:.3f}->{glog['sigma_tn']:.3f}  "
                    f"gnorm {float(gng) if gng is not None else float('nan'):.2f}  {sps:.2f} it/s")
                if _jsonl is not None:
                    _jsonl.log({"step": step, "cd": avg, "it_per_s": sps})
            if step % args.save_every == 0 or step == args.max_steps:
                # save BOTH the raw student (resume) AND the EMA (deploy target — infer with this).
                accel.wait_for_everyone()
                sd_g = accel.get_state_dict(student)
                sd_e = accel.get_state_dict(ema)
                if is_main:
                    import json
                    import safetensors.torch as st
                    dpath = runlib.dit_path(run_dir, step)                       # dits/dit_step{N} (raw student)
                    epath = os.path.join(run_dir, "ema", f"ema_step{step}.safetensors")  # DEPLOY weights
                    os.makedirs(os.path.dirname(dpath), exist_ok=True)
                    os.makedirs(os.path.dirname(epath), exist_ok=True)
                    st.save_file({k: v.contiguous() for k, v in sd_g.items()}, dpath)
                    st.save_file({k: v.contiguous() for k, v in sd_e.items()}, epath)
                    with open(os.path.join(run_dir, "progress.json"), "w") as f:
                        json.dump({"step": step}, f)
                    es = sorted(glob.glob(os.path.join(run_dir, "ema", "ema_step*.safetensors")),
                                key=lambda p: int(p.split("step")[-1].split(".")[0]))
                    for old in es[:-args.keep_last_states]:
                        os.remove(old)
                accel.wait_for_everyone()
                log(f"saved step {step}: dits/dit_step{step} (student) + ema/ema_step{step} (DEPLOY)")
    if _jsonl is not None:
        _jsonl.close()
    log("CD-INIT DONE  (deploy = ema/ema_step{last}.safetensors; feed to dmd_gm as --student_ckpt)")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        main()
