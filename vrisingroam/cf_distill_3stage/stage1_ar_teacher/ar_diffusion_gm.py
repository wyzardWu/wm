"""GameMaster Stage-1 for scheme-2 P1 — CAUSAL AR-diffusion (the CD teacher P1 needs).

Causal Forcing++ Stage-2 (CD, distill/cd_gm.py) requires a CAUSAL multi-step diffusion
teacher (师生同结构/同轨迹). Our existing Stage-1 is the BIDIRECTIONAL teacher gm_all3_v1,
which CF++ uses only later in DMD. So for P1 we first "causalize" the bidir teacher:
finetune a GameMasterDiT(causal=True), initialized from gm_all3_v1, on the flow-matching
objective under the block-causal self-attn mask.

v2 NOISING (DIFFUSION FORCING — the fix): each frame gets an INDEPENDENT per-frame noise
level (AR-order biased: later frames noisier), and frame0 (the i2v anchor) is clean most of
the time but occasionally small-noise corrupted (history-noise augmentation). The original v1
used ONE shared t for all frames 1..F-1 with a perfectly-clean frame0 — so it never trained
the (near-clean history, noisy current) regime that AR rollout visits, and free rollout
collapsed into grid/waxy noise within ~15s (proven). v2 makes that regime in-distribution.
See ar_diffusion_loss + gamemaster/flow_match.py sample_pf_timestep_ids/add_noise_pf.
The result is a causal multi-step AR-diffusion GameMaster = the P1 CD teacher (and student init).

This is distill/ode_init_gm.py's exact infrastructure MINUS the teacher: instead of aligning
the student velocity to a frozen teacher's velocity, we regress it to the true flow-matching
target v = noise - x0 (standard rectified-flow training, the Stage-1 objective). One trainable
model, no distillation, FSDP.

NOTE (P2 needs none of this): P2 reuses the bidir gm_all3_v1 directly as the CD teacher.
Run this ONLY for P1.

Launch (4-GPU accelerate FSDP):
    accelerate launch --config_file scripts/accelerate_fsdp.yaml distill/ar_diffusion_gm.py \
        --data data/precomputed/all3_F21 \
        --init_ckpt models/gm_all3_v1/dits/dit_step30000.safetensors \
        --lr 1e-5 --batch_size 1 --grad_accum 4 --max_steps 3000 \
        --out models/distill/gm_ar_causal_v1
Then feed dits/dit_step{N} to cd_gm.py as BOTH --teacher_ckpt and --student_ckpt with
--teacher_causal 1.
"""
import argparse
import glob
import os
import sys
import time

import torch

GM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GM)

from gamemaster.models.gamemaster_dit import GameMasterDiT, ti2v_5b_config, tiny_config
from gamemaster.flow_match import FlowMatchScheduler
from gamemaster.data.precomputed import PrecomputedRenderDataset, collate_precomputed
from gamemaster.infra import run as runlib


def load_dit_ckpt(model, ckpt_path):
    import safetensors.torch as st
    sd = st.load_file(ckpt_path)
    info = model.load_state_dict(sd, strict=False, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"ckpt mismatch: missing={info.missing_keys[:6]} unexpected={info.unexpected_keys[:6]}")
    return model


def ar_diffusion_loss(student, sched, latent, context, ar_order_bias=0.0,
                      hist_noise_prob=0.0, hist_noise_max=0.3,
                      teacher_forcing=False, tf_noise_aug_max_idx=0):
    """Causal flow-matching loss. TWO modes (backward-compatible):

    teacher_forcing=False (DEFAULT, UNCHANGED v3 DIFFUSION-FORCING): each frame 1..F-1 noised at
    an INDEPENDENT UNIFORM per-frame level (ar_order_bias=0), frame0 forced clean (i2v anchor,
    excluded from loss), per-frame bsmntw BELL weight (ReactiveGWM ar_tf.py:199/208-209/237). Knobs
    ar_order_bias>0 / hist_noise_prob>0 are our v2 additions (do NOT use bias>0 with the bell weight).

    teacher_forcing=True (CF/CF++ PARITY, the conditioning fix): history is a separate CLEAN-GT
    channel (clean_x), NOT a noised stream. Every frame (incl. frame0) of the NOISY stream is
    independently uniform-noised; the model denoises each noisy frame conditioned ONLY on the CLEAN
    prefix 0..f-1 (TF mask). Loss = bell-weighted MSE(v_pred, noise-clean) on ALL F frames (no frame0
    carve-out — the clean conditioning comes from clean_x, not from leaving frame0 clean in the noisy
    stream). tf_noise_aug_max_idx>0 enables CF's small history-noise augmentation on the clean channel
    (reuses the SAME noise; near-clean band = LOW sigma idx). =0 ⇒ exact clean (CF default, aug_t=None).
    Mirrors vendor/Causal-Forcing/model/diffusion.py:70-124 (clean_latent_aug :90-108, clean_x/aug_t
    call :112-118, loss+bell :120-124)."""
    B, C, Fl = latent.shape[:3]
    dev = latent.device

    if teacher_forcing:
        # ── TEACHER FORCING (CF parity) ──
        noise = torch.randn_like(latent)
        # noisy stream: every frame (incl. frame0) independent UNIFORM t (CF _get_timestep(0,N), nfpb=1)
        idx_pf = sched.sample_pf_timestep_ids(B, Fl, dev, ar_order_bias=0.0)             # [B,Fl] uniform
        noisy = sched.add_noise_pf(latent, noise, idx_pf)                                # ONLY this noised
        ts_pf = sched.timesteps.to(dev, torch.float32)[idx_pf]                           # [B,Fl] noisy-stream t

        # clean stream: exact GT (aug_t=None ⇒ t=0) OR small per-frame history-noise aug (CF, same noise)
        if tf_noise_aug_max_idx > 0:
            idx_aug = torch.randint(0, int(tf_noise_aug_max_idx), (B, Fl), device=dev)   # near-clean band (low idx)
            clean_x = sched.add_noise_pf(latent, noise, idx_aug)                          # reuse SAME noise (CF parity)
            aug_t = sched.timesteps.to(dev, torch.float32)[idx_aug]                       # [B,Fl] clean-stream t
        else:
            clean_x, aug_t = latent, None                                                # exact clean, t=0

        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            v = student(noisy, ts_pf, context, clean_x=clean_x, aug_t=aug_t)             # noisy-stream pred [B,C,Fl,H,W]
        target = (noise - latent)                                                        # v-target (rectified flow)
        se = (v.float() - target.float()).pow(2).mean(dim=(1, 3, 4))                      # [B,Fl]  ALL frames
        w = sched.training_weight(idx_pf).float()                                         # [B,Fl] per-frame bell
        return (w * se).mean()

    # ── DIFFUSION FORCING (UNCHANGED v3 path) ──
    # frames 0..F-1: independent per-frame noise levels (Diffusion Forcing, AR-order biased)
    idx_pf = sched.sample_pf_timestep_ids(B, Fl, dev, ar_order_bias=ar_order_bias)   # [B,Fl]
    # frame0 anchor: clean (prob 1-hist_noise_prob) OR small-noise sigma~U(0,hist_noise_max)
    noised0 = torch.rand(B, device=dev) < hist_noise_prob                            # [B] bool
    s0 = torch.rand(B, device=dev) * hist_noise_max                                  # [B] small sigma
    idx0_small = sched.sigma_to_idx(s0)                                              # [B] long
    clean_ph = idx_pf.new_full((B,), sched.num_train_timesteps - 1)                  # sigma~0 placeholder (frame0 overwritten + excluded below)
    idx_pf[:, 0] = torch.where(noised0, idx0_small, clean_ph)

    noise = torch.randn_like(latent)
    noisy = sched.add_noise_pf(latent, noise, idx_pf)                                # [B,C,Fl,H,W]
    # where frame0 is meant clean, overwrite with the exact clean latent + t=0
    clean0 = ~noised0                                                                # [B]
    f0 = (torch.arange(Fl, device=dev)[None, :] == 0) & clean0[:, None]              # [B,Fl]
    noisy = torch.where(f0.view(B, 1, Fl, 1, 1), latent, noisy)
    ts_pf = sched.timesteps.to(dev, torch.float32)[idx_pf]                           # [B,Fl] continuous t
    ts_pf[:, 0] = torch.where(clean0, ts_pf.new_zeros(B), ts_pf[:, 0])

    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        v = student(noisy, ts_pf, context)                  # causal student
    target = (noise - latent)                               # v-target (rectified flow)
    se = (v[:, :, 1:].float() - target[:, :, 1:].float()).pow(2).mean(dim=(1, 3, 4))  # [B,Fl-1]
    # REFERENCE-FAITHFUL (ReactiveGWM ar_tf.py:237): per-frame bsmntw bell weight on the MSE.
    # Correct ONLY because the default ar_order_bias=0 (uniform per-frame noise) — the bell weight's
    # high-sigma dead-zone then does NOT anticorrelate with the (uniform) sampling. If ar_order_bias>0
    # is ever set, the bell weight would silently starve the high-sigma regime (the v2 bug); keep
    # ar_order_bias=0 for parity with the reference.
    w = sched.training_weight(idx_pf[:, 1:]).float()                                  # [B,Fl-1] per-frame bell
    return (w * se).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(GM, "data/precomputed/all3_F21"))
    ap.add_argument("--v4_prompts", default=None,
                    help="v4.1 orthogonal filter output v4_prompts.pt (per-clip cleaned "
                         "player_nl/boss_nl + keep). Set with --v4_text_table to train on the "
                         "v4 clean orthogonal vocabulary (overrides move_sidecar path).")
    ap.add_argument("--v4_text_table", default=None,
                    help="v4 UMT5 text_table.pt (full-sentence emb) — REQUIRED with --v4_prompts.")
    ap.add_argument("--move_sidecar", default=None,
                    help="fine-prompt move sidecar (move_sidecar.pt); unset = coarse prompts")
    ap.add_argument("--coarse_frac", type=float, default=0.0)
    ap.add_argument("--balance", default="none", choices=["none", "game", "game_boss"],
                    help="rebalance train clip sampling across games/bosses (deterministic oversampling)")
    ap.add_argument("--balance_alpha", type=float, default=1.0)
    ap.add_argument("--balance_max_repeat", type=float, default=8.0)
    ap.add_argument("--init_ckpt", required=True, help="bidir gm_all3_v1 dit_step{N} to causalize")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=200)
    # Diffusion-Forcing knobs. v3 DEFAULTS = REFERENCE-FAITHFUL (ReactiveGWM ar_tf.py): bias=0 +
    # always-clean frame0 + (with the bell weight in the loss) = uniform independent per-frame noise.
    ap.add_argument("--ar_order_bias", type=float, default=0.0,
                    help="[DF mode only] 0=reference (pure independent UNIFORM per-frame noise; matches ar_tf.py:199). "
                         ">0 biases LATER frames noisier — our v2 addition; do NOT use with the bell "
                         "weight (it under-covers the clean sigma CD queries).")
    ap.add_argument("--hist_noise_prob", type=float, default=0.0,
                    help="[DF mode only] 0=reference (frame0 always clean). >0 = our v2 frame0 history-noise aug "
                         "(teaches error-correction of imperfect committed history).")
    ap.add_argument("--hist_noise_max", type=float, default=0.3,
                    help="[DF mode only] max sigma for the frame0 history-noise augmentation (U(0,this)).")
    # ── v4 TEACHER FORCING (CF/CF++ parity — the conditioning fix) ──
    ap.add_argument("--teacher_forcing", action=argparse.BooleanOptionalAction, default=True,
                    help="v4 DEFAULT ON: clean GT history as a clean_x channel; only the current (noisy) "
                         "stream is noised; the noisy frame attends ONLY the CLEAN prefix; loss on the "
                         "noisy stream over ALL frames (CF model/diffusion.py). --no-teacher_forcing "
                         "restores the v3 Diffusion-Forcing path (byte-identical to before).")
    ap.add_argument("--tf_noise_aug_max_idx", type=int, default=0,
                    help="[TF mode only] CF noise_augmentation_max_timestep analogue. 0=exact-clean history "
                         "(CF default, aug_t=None). >0 = small per-frame history-noise aug on clean_x "
                         "(near-clean band = LOW sigma idx; reuses the same noise). Keep small, e.g. <=20-50.")
    # ── windowed TRAINING toggle (train==infer windowed for long video; ship AFTER TF lands) ──
    ap.add_argument("--windowed_train", type=int, default=0,
                    help="0=OFF (full-causal training mask, UNCHANGED). >0 = trailing window of this many "
                         "frames on the TRAINING self-attn mask (CF long_video local_attn_size; RoPE stays "
                         "absolute). Isolate the TF conditioning fix FIRST (window OFF, <=21 frames), then "
                         "enable for long rollout.")
    ap.add_argument("--windowed_sink", type=int, default=0,
                    help="[windowed_train only] number of always-visible early (sink) frames (CF sink_size; "
                         "released CF config uses 0).")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--keep_last_states", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader

    accel = Accelerator(gradient_accumulation_steps=args.grad_accum, step_scheduler_with_optimizer=False)
    set_seed(args.seed, device_specific=True)   # per-rank RNG so Diffusion-Forcing draws differ across ranks
    is_main = accel.is_main_process
    run_dir = runlib.resolve_run_dir(args.out)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        runlib.save_config(run_dir, {**vars(args), "world_size": accel.num_processes, "stage": "ar_diffusion_gm"})
        _tee = runlib.TextTee(os.path.join(run_dir, "train.log"))
        _jsonl = runlib.JsonlLogger(os.path.join(run_dir, "train_log.jsonl"))
        log = _tee
    else:
        _jsonl = None
        def log(*a):
            pass
    log(f"=== GM causal AR-diffusion (scheme2 P1 Stage-1) | run={run_dir} world={accel.num_processes} "
        f"lr={args.lr} bs={args.batch_size} accum={args.grad_accum} "
        f"teacher_forcing={args.teacher_forcing} tf_noise_aug_max_idx={args.tf_noise_aug_max_idx} "
        f"windowed_train={args.windowed_train} windowed_sink={args.windowed_sink} ===")

    cfg = tiny_config() if args.dry_run else ti2v_5b_config()
    student = GameMasterDiT(**cfg, causal=True, zero_init_head=False,
                            train_local_attn_size=args.windowed_train,
                            train_sink_size=args.windowed_sink)
    if not args.dry_run:
        log("loading: causal student <- bidir gm_all3_v1 (causalize) ...")
        load_dit_ckpt(student, args.init_ckpt)
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
        log(f"clips: {len(all_files)} -> {len(train_files)} train; dataset {len(ds)}")
    else:
        class _Syn(torch.utils.data.Dataset):
            def __len__(self): return 64
            def set_epoch(self, e): self.epoch = e
            epoch = 0
            def __getitem__(self, i):
                g = torch.Generator().manual_seed(i)
                return {"latent": torch.randn(48, 5, 8, 8, generator=g, dtype=torch.bfloat16),
                        "context": torch.randn(5, 4, 4096, generator=g, dtype=torch.bfloat16),
                        "boss_dropped": False, "fight": 0, "start_cell": i}
        ds = _Syn()
        if args.max_steps > 30:
            args.max_steps, args.save_every, args.warmup, args.log_every = 12, 6, 3, 2
        log("DRY RUN: tiny model, synthetic data")

    student.use_gradient_checkpointing = True
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate_precomputed, drop_last=True, pin_memory=True)
    # vendor parity (ReactiveGWM CF upstream, default.yaml:74-78): ar_beta1=0.0 ar_beta2=0.999 wd=0.01
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.0, 0.999))

    def lr_lambda(s):
        return (s + 1) / args.warmup if s < args.warmup else 1.0
    sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    class _Counter:
        def __init__(self): self.step = 0
        def state_dict(self): return {"step": self.step}
        def load_state_dict(self, sd): self.step = int(sd["step"])
    counter = _Counter()
    accel.register_for_checkpointing(counter)

    student, opt, dl, sched_lr = accel.prepare(student, opt, dl, sched_lr)
    sched = FlowMatchScheduler(shift=5.0).to(accel.device)

    if args.resume:
        if args.resume == "auto":
            lt = runlib.latest_state(run_dir)            # None if no saved state yet
            rp = lt[1] if isinstance(lt, tuple) else lt
        else:
            rp = args.resume
        if rp:
            accel.load_state(rp); log(f"resumed from {rp} at step {counter.step}")
        else:
            log("--resume auto but no saved state in run_dir; starting fresh")
    step = counter.step
    _ls, _lt, _ema = step, time.time(), [None]

    student.train()
    data_iter = iter(dl)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            ds.set_epoch(getattr(ds, "epoch", 0) + 1); data_iter = iter(dl); batch = next(data_iter)
        with accel.accumulate(student):
            latent = batch["latent"].to(accel.device)
            context = batch["context"].to(accel.device)
            loss = ar_diffusion_loss(student, sched, latent, context,
                                     ar_order_bias=args.ar_order_bias,
                                     hist_noise_prob=args.hist_noise_prob,
                                     hist_noise_max=args.hist_noise_max,
                                     teacher_forcing=args.teacher_forcing,
                                     tf_noise_aug_max_idx=args.tf_noise_aug_max_idx)
            accel.backward(loss)
            gn = None
            if accel.sync_gradients:
                gn = accel.clip_grad_norm_(student.parameters(), 10.0)  # vendor parity: ar_grad_clip=10.0 (default.yaml:79)
            opt.step(); opt.zero_grad()
        if accel.sync_gradients:
            sched_lr.step(); step += 1; counter.step = step
            if step % args.log_every == 0:
                avg = accel.gather(loss.detach()).mean().item()
                now = time.time(); sps = (step - _ls) / max(now - _lt, 1e-6); _ls, _lt = step, now
                _ema[0] = avg if _ema[0] is None else 0.9 * _ema[0] + 0.1 * avg
                log(f"step {step:6d}/{args.max_steps}  ar_v {avg:.4f} (ema {_ema[0]:.4f})  "
                    f"lr {sched_lr.get_last_lr()[0]:.2e}  gnorm {float(gn) if gn is not None else float('nan'):.2f}  {sps:.2f} it/s")
                if _jsonl is not None:
                    _jsonl.log({"step": step, "ar_v": avg, "ema": _ema[0], "it_per_s": sps})
            if step % args.save_every == 0 or step == args.max_steps:
                accel.save_state(runlib.state_dir(run_dir, step))
                accel.wait_for_everyone()
                sd = accel.get_state_dict(student)
                if is_main:
                    import safetensors.torch as st
                    dpath = runlib.dit_path(run_dir, step)
                    os.makedirs(os.path.dirname(dpath), exist_ok=True)
                    st.save_file({k: v.contiguous() for k, v in sd.items()}, dpath)
                    runlib.rotate_states(run_dir, args.keep_last_states)
                accel.wait_for_everyone()
                log(f"saved step {step}: dit -> dits/dit_step{step}.safetensors")
    if _jsonl is not None:
        _jsonl.close()
    log("CAUSAL AR-DIFFUSION DONE  (feed dits/dit_step{last} to cd_gm.py as teacher+student, --teacher_causal 1)")


if __name__ == "__main__":
    main()
