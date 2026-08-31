"""GameMaster Stage-2b — Self-Forcing DMD (Option A, NATIVE conventions throughout).

Anti-bug rule (the prior failure was driving our vanilla-bidir teacher through a streaming
forward): teacher(real_score) and critic(fake_score) are NATIVE GameMasterDiT(causal=False)
— full bidir, consecutive RoPE, C9 per-frame text, frame0-clean; the student is
GameMasterDiT(causal=True) with the KV-cache rollout (proven == full-clip causal). DMD scores
the WHOLE generated clip with the SAME native TI2V noising the teacher was SFT'd on, so the
teacher stays in-distribution. Vendored DMD MATH is the reference; its streaming conventions
are dropped.

Core pieces (verified by --smoke before any 5B/FSDP run):
  self_forcing_rollout  — student generates a clip autoregressively via forward_frame
  dmd_generator_loss    — distribution-matching gradient (teacher CFG vs critic) -> student
  critic_loss           — critic learns to flow-denoise the student's own samples
The 5 alignment asserts run on every teacher/critic forward as a fail-fast gate.
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


# ───────────────────────── alignment gate (the "don't repeat the bug" asserts) ─────────────────────────

def _assert_native_scoring(latent, ts_pf, context):
    """Fail-fast before any teacher/critic forward. Asserts 1&2 (no frame_ids / no streaming
    self_mask) are guaranteed by construction — GameMasterDiT has no such params. These guard
    the INPUTS: channels-first, C9 per-frame context, frame0-clean + frames 1..N share one t."""
    B, C, Fl = latent.shape[:3]
    assert C == 48, f"[gate] channels-first [B,48,F,H,W] required, got {tuple(latent.shape)}"
    assert context.dim() == 4 and context.shape[1] == Fl, \
        f"[gate] C9 per-frame context [B,F,512,4096] required, got {tuple(context.shape)}"
    assert (ts_pf[:, 0] == 0).all(), "[gate] frame0 must be clean (t=0)"
    assert torch.allclose(ts_pf[:, 1:], ts_pf[:, 1:2].expand_as(ts_pf[:, 1:])), \
        "[gate] frames 1..N must share one sampled timestep (native TI2V)"


def _sample_dmd_timestep(B, sched, dev, gen=None):
    """One shared timestep id per sample (the scheduler timesteps are already shift-warped);
    clamp to [0.02,0.98]*1000 -> indices via nearest."""
    n = sched.timesteps.shape[0]
    lo, hi = int(0.02 * n), int(0.98 * n)
    return torch.randint(lo, hi, (B,), device=dev, generator=gen)


def _ti2v_noise(clip, sched, dev, gen=None):
    """Native TI2V noising == ode_init_gm / scripts/train.py: frame0 clean (t=0), frames 1..F-1
    noised at ONE shared sampled t. Returns (noisy, ts_pf[B,F], sigma[B], idx[B])."""
    B, C, Fl = clip.shape[:3]
    idx = _sample_dmd_timestep(B, sched, dev, gen)
    noise = torch.randn(clip.shape, device=dev, dtype=clip.dtype, generator=gen)
    noisy = sched.add_noise(clip, noise, idx)
    ts_pf = sched.timesteps.to(dev, torch.float32)[idx][:, None].expand(B, Fl).clone()
    ts_pf[:, 0] = 0.0
    f0 = (torch.arange(Fl, device=dev)[None, :] == 0).view(1, 1, Fl, 1, 1)
    noisy = torch.where(f0, clip, noisy)                      # frame0 = clean
    sigma = sched.sigmas.to(dev, torch.float32)[idx]          # [B]
    return noisy, ts_pf, sigma, noise


# ───────────────────────── self-forcing rollout (student generates a clip) ─────────────────────────

def self_forcing_rollout(student, initial_latent, prompt_embeds, num_gen_frames,
                         denoising_steps, grad_horizon, sched, gen=None,
                         kv_window=17, rope_cap=20, sink_size=3, score_last_k=None,
                         vae=None, reanchor_every=16, reanchor_win=6):
    """Student autoregressively generates `num_gen_frames` frames from `initial_latent` via the
    KV-cache forward, conditioning on its OWN committed frames (self-forcing). Only a window of
    `grad_horizon` frames retains autograd (bounds memory).
      initial_latent: [B,C,1,H,W] real clean start (clip frame 0)
      prompt_embeds:  [B, 1+N, L, 4096] FULL-clip per-frame prompts (frame k uses [:,k])  (C9)
    WINDOWED (inference-aligned) rollout: the KV cache is built with kv_window/rope_cap/sink_size
    so the deployed sliding geometry (eviction past sink+kv_window) is actually trained. The
    sliding read (gamemaster_dit.py:201) is ON the autograd graph (only the COMMITTED cache write
    at :208-209 is .detach()ed, gated by commit=True) -> grad flows through the windowed transient
    (commit=False) pass exactly as in the absolute path. BACKWARD-COMPATIBLE: kv_window<=0 falls
    back to the OLD bare init_kv_cache() (absolute/unbounded), zero behavior change.
    score_last_k: if set, PIN the grad window to the last K generated frames (tail), so autograd
    frames land on post-eviction frames (CF base.py:154-167); else use the OLD random window.
    VAE RE-ANCHOR (CF base.py:154-167, Rolling-Forcing): if vae is given and reanchor_every>0,
    every R=reanchor_every committed frames the committed-as-context latent is projected back onto
    the VAE manifold (decode a W_dec=reanchor_win-latent WINDOW -> keep ONLY the last pixel frame ->
    re-encode that 1 frame -> re-commit it DETACHED), so the conditioning never drifts off-manifold
    over a long rollout (the darken/wash root cause). The whole op is under no_grad + .detach() ==
    CF gradient_mask[:, :block]=False (re-anchored frame is DETACHED context, never a scored grad
    frame). BACKWARD-COMPATIBLE: reanchor_every<=0 (or vae=None) reproduces the OLD no-re-anchor path.
    Returns: generated [B,C,1+N,H,W] (frame0=real start), grad_mask [1+N] bool."""
    B, C, _, Hh, Ww = initial_latent.shape
    dev = initial_latent.device
    if vae is not None:
        assert B == 1, f"[re-anchor] decode/encode hardcodes batch index 0; require B==1, got {B} (BUG-3)"
    total = 1 + num_gen_frames
    nsteps = len(denoising_steps)
    sig = sched.sigmas.to(dev, torch.float32)
    tsteps = sched.timesteps.to(dev, torch.float32)

    def _sigma_of(tsv):
        return sig[int(torch.argmin((tsteps - float(tsv)).abs()))]

    # init_kv_cache lives on the bare module (FSDP wraps forward(), not custom methods)
    bare = student.module if hasattr(student, "module") else student
    # WINDOWED (inference-aligned): pass kv_window/rope_cap/sink_size so the sliding geometry is
    # trained. BACKWARD-COMPATIBLE: kv_window<=0 (or unset via the caller) -> OLD bare/absolute call.
    if kv_window is not None and kv_window > 0:
        cache = bare.init_kv_cache(kv_window=kv_window, rope_cap=rope_cap, sink_size=sink_size)
    else:
        cache = bare.init_kv_cache()
    # X-Cache (cross-frame block-residual reuse): built ONLY when GM_XCACHE=1; else None ⇒ every
    # student() forward below is byte-identical to today. CAUTION: X-Cache stores DETACHED residuals,
    # so a cache HIT on a grad-carrying frame would cut autograd through that block. This is an
    # INFERENCE-ONLY accelerator — do NOT set GM_XCACHE during DMD training. Default (unset) is safe.
    xcache = bare.init_xcache() if os.environ.get("GM_XCACHE") == "1" else None
    # commit frame0 (real clean start) as context — its own prompt, t=0.
    # NB: call via forward(..., kv_cache=) so FSDP's param all-gather fires (not forward_frame directly).
    with torch.no_grad():
        student(initial_latent, torch.zeros(B, device=dev),
                prompt_embeds[:, 0:1], frame_index=0, kv_cache=cache, commit=True)
    frame_list = [initial_latent[:, :, 0:1].detach()]

    # grad window + random exit step (same_step_across_blocks).
    # When score_last_k is set, BIAS the (still grad_horizon-wide) window INTO the scored tail so
    # the autograd frames are scored frames (else dmd eval_idx ∩ grad_mask is empty -> silent 0 loss,
    # see RISK in dmd_generator_loss). Tail span = last K generated frames = [1+N-K, total).
    if score_last_k is not None:
        tail_start = max(1, total - score_last_k)               # first scored generated frame
        lo = tail_start                                          # grad_start in [tail_start, total-grad_horizon]
        hi = max(tail_start, total - grad_horizon)
        grad_start = int(torch.randint(lo, hi + 1, (1,), device=dev, generator=gen).item())
    else:
        max_grad_start = max(1, total - grad_horizon)           # OLD random window over the whole rollout
        grad_start = int(torch.randint(1, max_grad_start + 1, (1,), device=dev, generator=gen).item())
    grad_end = grad_start + grad_horizon
    exit_step = int(torch.randint(0, nsteps, (1,), device=dev, generator=gen).item())

    for k in range(1, total):
        ctx_k = prompt_embeds[:, k:k + 1]
        noisy = torch.randn(B, C, 1, Hh, Ww, device=dev, dtype=initial_latent.dtype, generator=gen)
        in_window = (grad_start <= k < grad_end)
        den = None
        for si, tsv in enumerate(denoising_steps):
            is_exit = (si == exit_step)
            grad_on = in_window and is_exit
            ts_k = torch.full((B,), float(tsv), device=dev)
            with (torch.enable_grad() if grad_on else torch.no_grad()):
                v = student(noisy, ts_k, ctx_k, frame_index=k, kv_cache=cache, commit=False,
                            step_index=si, xcache=xcache)
                den = noisy - _sigma_of(tsv) * v               # x0 estimate
            if is_exit:
                break
            nxt = denoising_steps[si + 1]                      # re-noise x0 -> next level (detached)
            ns = _sigma_of(nxt)
            noisy = (1 - ns) * den.detach() + ns * torch.randn_like(den)
        frame_list.append(den)                                # carries grad iff in_window
        # ── VAE RE-ANCHOR (CF base.py:154-167): on a re-anchor frame, COMMIT THE MANIFOLD ANCHOR
        #    *INSTEAD OF* the raw few-step den (NOT in addition — the sliding commit is APPEND-only
        #    [gamemaster_dit.py:203 torch.cat], so a second commit at the same frame_index doubles
        #    frame k in the cache, inflating n_cached and corrupting RoPE(f_start=tpos-R)/eviction for
        #    the rest of the rollout: BUG-1). Decode a WINDOW (our Wan2.2 VAE is causal/4x-temporal ->
        #    a 1-latent decode is NOT identity & injects drift), keep ONLY the last fully-receptive
        #    pixel frame, re-encode -> ONE manifold latent, commit it DETACHED (gradient-masked
        #    context == CF gradient_mask[:, :block]=False). channels-first throughout (our vae list API
        #    drops batch -> [C,F,H,W], no transpose). GUARD: only on non-grad frames (not in_window)
        #    so we never replace a grad-carrying scored frame with a detached anchor. ──
        do_reanchor = (vae is not None and reanchor_every > 0 and (k % reanchor_every == 0)
                       and k >= reanchor_win and not in_window)
        with torch.no_grad():
            if do_reanchor:
                win = torch.cat(frame_list[k - reanchor_win + 1: k + 1], dim=2)   # [B,48,Wd,h,w] committed clean
                pix = vae.decode([win[0].float()])[0]                             # [3,T,H,W] T=4*Wd-3 (rollout.py:144)
                last_pix = pix[:, -1:, :, :]                                       # [3,1,H,W] KEEP LAST PIXEL FRAME ONLY
                anchor = vae.encode([last_pix])[0].unsqueeze(0).to(initial_latent.dtype).detach()  # [B,48,1,h,w]
                frame_list[k] = anchor                                            # output[k] + committed context both = manifold latent
                commit_lat = anchor
            else:
                commit_lat = den.detach()
            student(commit_lat, torch.zeros(B, device=dev), ctx_k,               # commit EXACTLY ONCE per frame_index
                    frame_index=k, kv_cache=cache, commit=True)

    generated = torch.cat(frame_list, dim=2)                  # [B,C,1+N,H,W]
    grad_mask = torch.tensor([grad_start <= k < grad_end for k in range(total)], device=dev)
    return generated, grad_mask


# ───────────────────────── DMD generator loss (distribution matching) ─────────────────────────

def dmd_generator_loss(teacher, critic, generated, prompt_embeds, uncond_embeds,
                       grad_mask, sched, real_guidance_scale=3.0, gen=None):
    """DMD: push the student's generated frames toward the teacher's distribution.
    Noise the FULL generated clip TI2V-style; teacher(CFG) + critic score it (native bidir);
    g = pred_fake - pred_real (x0 space); 0.5*MSE surrogate whose grad == normalized g, applied
    on the grad-window frames (which carry student autograd)."""
    B, C, Fl = generated.shape[:3]
    dev = generated.device
    eval_idx = torch.nonzero(grad_mask, as_tuple=False).flatten()
    eval_idx = eval_idx[eval_idx >= 1]                        # never frame0
    with torch.no_grad():
        noisy, ts_pf, sigma, _ = _ti2v_noise(generated.detach(), sched, dev, gen)
        sig = sigma.view(B, 1, 1, 1, 1)
        _assert_native_scoring(generated, ts_pf, prompt_embeds)
        # teacher real-score. CFG=1.0 IRON RULE (gm-cfg-no-dropout): the bidir teacher was SFT'd
        # WITHOUT text-dropout, so the null-text (uncond) forward is OOD. At scale==1.0 use cond-only
        # and SKIP the uncond forward entirely (mirrors cd_gm.py); only apply standard CFG for scale>1.
        v_real_c = teacher(noisy, ts_pf, prompt_embeds)
        x0_real_c = noisy - sig * v_real_c
        if abs(real_guidance_scale - 1.0) < 1e-6:
            pred_real = x0_real_c                             # cond-only (no OOD null-text forward)
        else:
            x0_real_u = noisy - sig * teacher(noisy, ts_pf, uncond_embeds)
            pred_real = x0_real_u + real_guidance_scale * (x0_real_c - x0_real_u)  # standard CFG
        # critic fake-score
        v_fake = critic(noisy, ts_pf, prompt_embeds)
        pred_fake = noisy - sig * v_fake
        # DMD gradient in x0 space (per-frame), normalized per-sample over eval frames
        raw_grad = pred_fake - pred_real                      # [B,C,F,H,W]
        p_real = generated.detach() - pred_real
        normalizer = p_real[:, :, eval_idx].abs().mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-2)
        grad = torch.nan_to_num(raw_grad / normalizer)
    # 0.5*MSE surrogate: d/d generated == grad, only on grad-window frames
    target = (generated - grad).detach()
    g = generated[:, :, eval_idx].float()
    t = target[:, :, eval_idx].float()
    loss = 0.5 * F.mse_loss(g, t)
    return loss, {"dmd_grad_norm": grad[:, :, eval_idx].norm().item()}


# ───────────────────────── critic loss (fake-score learns the student distribution) ─────────────────────────

def critic_loss(critic, generated, prompt_embeds, sched, gen=None):
    """Critic learns to flow-denoise the student's OWN generated frames (native flow-matching),
    so its score is a valid fake-distribution score for the DMD term. Generation is detached."""
    B, C, Fl = generated.shape[:3]
    dev = generated.device
    clip = generated.detach()
    noisy, ts_pf, sigma, noise = _ti2v_noise(clip, sched, dev, gen)
    _assert_native_scoring(clip, ts_pf, prompt_embeds)
    v_fake = critic(noisy, ts_pf, prompt_embeds)             # flow prediction
    target = noise - clip                                     # v = eps - x0 (x0 = generated)
    loss = F.mse_loss(v_fake[:, :, 1:].float(), target[:, :, 1:].float())   # frames 1..F-1
    return loss, {"critic_loss": loss.item()}


# ───────────────────────── CPU smoke (verify shapes + grad flow + math, no GPU/FSDP) ─────────────────────────

def smoke():
    torch.manual_seed(0)
    cfg = tiny_config()
    dev = "cpu"
    teacher = GameMasterDiT(**cfg, causal=False, zero_init_head=False).eval().requires_grad_(False).to(dev)
    critic = GameMasterDiT(**cfg, causal=False, zero_init_head=False).to(dev)
    student = GameMasterDiT(**cfg, causal=True, zero_init_head=False).to(dev)
    critic.load_state_dict(teacher.state_dict())
    student.load_state_dict(teacher.state_dict())
    sched = FlowMatchScheduler(shift=5.0).to(dev)

    B, C, N, Hp, Wp, L = 1, 48, 5, 8, 8, 4
    total = 1 + N
    initial = torch.randn(B, C, 1, Hp, Wp)
    prompts = torch.randn(B, total, L, 4096)
    uncond = torch.randn(B, total, L, 4096)
    denoising_steps = [1000, 500]

    print("── self-forcing rollout ──")
    generated, grad_mask = self_forcing_rollout(student, initial, prompts, N, denoising_steps,
                                                grad_horizon=3, sched=sched)
    print(f"generated {tuple(generated.shape)} (expect [1,48,{total},{Hp//2*2}, {Wp//2*2}]); grad_mask={grad_mask.tolist()}")
    assert generated.shape == (B, C, total, Hp, Wp)
    assert generated.requires_grad, "generated must carry student grad (grad window)"
    assert grad_mask.sum() > 0 and not grad_mask[0]

    print("── DMD generator loss + backward ──")
    gl, glog = dmd_generator_loss(teacher, critic, generated, prompts, uncond, grad_mask, sched)
    gl.backward()
    gp = [p for p in student.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"dmd loss={gl.item():.5f}  {glog}; student params with grad: {len(gp)} (must be >0)")
    assert len(gp) > 0, "DMD loss did not reach student params!"
    # teacher/critic must NOT have grads from the DMD loss (scored under no_grad)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in teacher.parameters())

    print("── critic loss + backward ──")
    student.zero_grad(); critic.zero_grad()
    cl, clog = critic_loss(critic, generated, prompts, sched)
    cl.backward()
    cp = [p for p in critic.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"critic loss={cl.item():.5f}; critic params with grad: {len(cp)} (must be >0)")
    assert len(cp) > 0, "critic loss did not reach critic params!"

    print("\n✅ DMD core OK: rollout generates a clip, DMD grad reaches the student (teacher frozen), "
          "critic grad reaches the critic. Alignment gate (C9/frame0-clean/shared-t/channels-first) passed.")


# ───────────────────────── trainer (Stage 2b) — 3-model FSDP, alternating critic/generator ─────────────────────────

def load_dit_ckpt(model, ckpt_path):
    import safetensors.torch as st
    sd = st.load_file(ckpt_path)
    info = model.load_state_dict(sd, strict=False, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"ckpt mismatch {ckpt_path}: missing={info.missing_keys[:6]} unexpected={info.unexpected_keys[:6]}")
    return model


@torch.no_grad()
def ema_update(ema, student, decay):
    """Per-shard generator EMA (CF++/ReactiveGWM deploy the EMA; Stage-2 CD does too). With
    fsdp_use_orig_params=true both models expose the SAME original Parameters in identical order,
    sharded identically -> zip is exact. decay<=0 => hard copy. (cd_gm.py:ema_update mirror.)"""
    for pe, p in zip(ema.parameters(), student.parameters()):
        if decay <= 0.0:
            pe.data.copy_(p.data)
        else:
            pe.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def main():
    GMroot = GM
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(GMroot, "data/precomputed/all3_F21"))
    ap.add_argument("--move_sidecar", default=None,
                    help="fine-prompt move sidecar (move_sidecar.pt); unset = coarse prompts")
    ap.add_argument("--coarse_frac", type=float, default=0.0)
    ap.add_argument("--balance", default="none", choices=["none", "game", "game_boss"],
                    help="rebalance train clip sampling across games/bosses (deterministic oversampling)")
    ap.add_argument("--balance_alpha", type=float, default=1.0)
    ap.add_argument("--balance_max_repeat", type=float, default=8.0)
    ap.add_argument("--teacher_ckpt", required=True, help="gm_all3_v1 dit_step{N}.safetensors (real-score + critic init)")
    ap.add_argument("--student_ckpt", required=True, help="gm_ode_v2 ODE-init dit_step{N}.safetensors (generator init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-6, help="generator (student) lr (ref 5e-6)")
    ap.add_argument("--lr_critic", type=float, default=1e-6, help="critic (fake-score) lr (ref 1e-6; was reversed)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--num_denoise_steps", type=int, default=2, help="DEPRECATED (use --denoise_list); kept for logs")
    ap.add_argument("--denoise_list", default="1000,250",
                    help="explicit raw t-value few-step schedule, ending LOW. 2-step [1000,250]=final sigma 0.25 "
                         "(ref 4-step [1000,750,500,250] same endpoint). REPLACES set_timesteps(2)=[1000,833] which "
                         "read x0 at sigma 0.833 => blur. MUST match inference.")
    ap.add_argument("--grad_horizon", type=int, default=14,
                    help="frames retaining autograd in the rollout (was 3; widened toward the scored ~21-frame "
                         "tail per CF rolling_forcing_training.py:124 start_gradient_frame_index=num_output-21; "
                         "kept <= dmd_score_frames so the grad window stays inside the scored region + activation budget)")
    ap.add_argument("--num_gen_frames", type=int, default=None,
                    help="BACK-COMPAT fixed-length override. If set, N=num_gen_frames (NO LONGER capped at "
                         "Fl-1; pass <=Fl-1 to reproduce old behavior). If None, N is sampled in "
                         "[rollout_min, rollout_max] per step (rank-synced).")
    # ── windowed (inference-aligned) DMD rollout (BACKWARD-COMPATIBLE: kv_window<=0 -> OLD absolute/unbounded) ──
    ap.add_argument("--kv_window", type=int, default=17,
                    help="sliding KV window; bounds the rollout cache (inference-aligned). Paired with sink_size=3 "
                         "so sink+window=20 -> read(commit=False)=20 cached+1 self=21 keys at RoPE pos 0..20 == "
                         "the AR's EXACT trained F=21 envelope (BUG-2: 18+3=21 was 1 key/1 pos OOD). "
                         "<=0 => OLD absolute/unbounded init_kv_cache() behavior (back-compat).")
    ap.add_argument("--rope_cap", type=int, default=20, help="RoPE position cap = F-1; steady-state sliding-path tpos clamps to 20 -> contiguous pos 0..20 == AR F=21 envelope (cap=21 left a gap at pos 3 + reached pos 21, 1 past AR's trained max)")
    ap.add_argument("--sink_size", type=int, default=3,
                    help="always-visible early (sink) frames. 3 = CF sink=1-block=3-frames; sink(3)+window(17)=20. "
                         "Was 1. frame0 is a REAL VAE-encoded latent -> a permanent on-manifold anchor.")
    ap.add_argument("--rollout_min", type=int, default=26,
                    help="min total generated frames N (>=F to exercise eviction); used when --num_gen_frames is None")
    ap.add_argument("--rollout_max", type=int, default=100,
                    help="max total generated frames N (was 41; bumped to ~100 to distill depth-robustness at the "
                         "deployed minute-scale horizon, ceil(100/16)=6 re-anchors/rollout). NOT CF's 101.")
    ap.add_argument("--dmd_score_frames", type=int, default=26,
                    help="K = restrict the DMD/critic score window to the last K generated frames (last-K, CF base.py:154-167)")
    ap.add_argument("--reanchor_every", type=int, default=16,
                    help="R: VAE re-anchor every R committed frames (CF base.py:154-167). Projects the committed "
                         "context latent back onto the VAE manifold (decode window -> keep last pixel frame -> "
                         "re-encode -> re-commit DETACHED). <=0 disables (OLD no-re-anchor, back-compat).")
    ap.add_argument("--reanchor_win", type=int, default=6,
                    help="W_dec: # committed latents decoded for the re-anchor window so the causal/4x-temporal "
                         "Wan2.2 VAE has full receptive field for the LAST pixel frame (6 latents -> 4*6-3=21 px "
                         "frames; keep only the last). NEVER decode a single latent (not identity -> injects drift).")
    ap.add_argument("--critic_steps", type=int, default=4,
                    help="critic updates per generator step = the critic:gen lead (ref dfake_gen_update_ratio=5, "
                         "CF++ trainer/distillation.py:307; default was 1 => 1:1 => fake-score never tracked the "
                         "moving student => biased mean-reverting DMD gradient => blur + flat loss). The generator "
                         "MUST update every step (FSDP drops the student grad if a student rollout is not immediately "
                         "followed by a student backward — test_fsdp_rollout_grad.py), so the lead comes from inner "
                         "critic steps on the SAME detached sample. CAVEAT: CF++ gives the critic 5 FRESH rollouts; "
                         "we reuse one (different noise/timestep per step) — the fresh-rollout diversity is deferred "
                         "(FSDP-blocked), so this knob matches CF++'s ratio but not its sample diversity.")
    ap.add_argument("--real_guidance_scale", type=float, default=1.0,
                    help="CFG on the teacher real-score. Ref uses 3.0 BUT our teacher has no text dropout "
                         "(scripts/train.py) => x0_cond≈x0_uncond, so 3.0 amplifies the near-zero noise residual "
                         "into character artifacts. 1.0 (mild) is the gm_dmd_v2 value; 0.0 = pure x0_cond (cleanest).")
    ap.add_argument("--opt8bit", type=int, default=1, help="1=bitsandbytes AdamW8bit (fit 3x5B), 0=fp32 AdamW")
    ap.add_argument("--paged_opt", type=int, default=0, help="1=page 8-bit opt state to CPU (doesn't cut the activation peak)")
    ap.add_argument("--profile_mem", type=int, default=0, help="1=log per-phase peak GPU mem on the first steps")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--keep_last_states", type=int, default=2)
    ap.add_argument("--use_ema", type=int, default=1,
                    help="1=maintain+deploy a generator EMA (4th FSDP model; CF++/ReactiveGWM/our CD do). "
                         "0=3-model DMD (no EMA) — comfortable on 3 cards + frees a card for parallel eval/diag.")
    ap.add_argument("--ema_decay", type=float, default=0.99,
                    help="generator EMA decay (CF++/ReactiveGWM ema_weight 0.99). DMD deploys the EMA "
                         "(both refs + our Stage-2 CD do); raw student kept separately for resume.")
    ap.add_argument("--ema_start", type=int, default=200,
                    help="hard-copy EMA<-student until this step, then decay (CF ema_start_step 200)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    assert args.rope_cap == args.sink_size + args.kv_window, (
        f"rope_cap ({args.rope_cap}) must equal sink_size + kv_window "
        f"({args.sink_size}+{args.kv_window}): re-RoPE'd cache positions must exactly fill "
        f"0..rope_cap or you get silent position holes / OOD positions past the trained envelope")

    import glob
    import time
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    sys.path.insert(0, GMroot)
    from gamemaster.data.precomputed import PrecomputedRenderDataset, collate_precomputed
    from gamemaster.infra import run as runlib

    accel = Accelerator(step_scheduler_with_optimizer=False)
    set_seed(args.seed)
    is_main = accel.is_main_process
    run_dir = runlib.resolve_run_dir(args.out)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        runlib.save_config(run_dir, {**vars(args), "world_size": accel.num_processes, "stage": "dmd_gm"})
        _tee = runlib.TextTee(os.path.join(run_dir, "train.log"))
        _jsonl = runlib.JsonlLogger(os.path.join(run_dir, "train_log.jsonl"))
        log = _tee
    else:
        _jsonl = None
        def log(*a):
            pass
    log(f"=== GM DMD (Stage 2b, native Option A) | run={run_dir} world={accel.num_processes} "
        f"lr_g={args.lr} lr_c={args.lr_critic} denoise={args.num_denoise_steps} "
        f"grad_h={args.grad_horizon} critic_steps={args.critic_steps} cfg={args.real_guidance_scale} ===")

    cfg = tiny_config() if args.dry_run else ti2v_5b_config()
    # 3 models: generator(causal,FSDP), critic(bidir,FSDP), teacher(bidir,frozen,replicated)
    student = GameMasterDiT(**cfg, causal=True, zero_init_head=False)
    critic = GameMasterDiT(**cfg, causal=False, zero_init_head=False)
    teacher = GameMasterDiT(**cfg, causal=False, zero_init_head=False)
    # generator EMA shadow (causal, frozen) — the DEPLOY weights (CF++/ReactiveGWM + our CD all
    # deploy the EMA; raw DMD deploying the drifting student is the leading darkening-symptom suspect).
    ema = GameMasterDiT(**cfg, causal=True, zero_init_head=False) if args.use_ema else None
    # Critic does a full 21-frame forward+backward each step (~15GB activation) -> checkpoint it.
    # Teacher is no_grad (no activation). Student's grad path is forward_frame (per-frame, tiny).
    critic.use_gradient_checkpointing = True
    student.use_gradient_checkpointing = True

    # few-step schedule: EXPLICIT raw t-value list ending at LOW noise (reference recipe
    # streaming_distill_5b_single.yaml: denoising_step_list [1000,750,500,250], warp=false).
    # ★ FIX: the old `set_timesteps(num_denoise_steps).infer_timesteps` gave [1000,833] for
    # 2-step under shift=5 — the FINAL x0 was read at sigma=0.833 (83% noise), so even a
    # perfect velocity field returns the conditional MEAN = intrinsically soft/dim/blurry
    # (confirmed: decoding the teacher's own x0 at sigma 0.833 is soft). _sigma_of maps each
    # raw t -> sigma via argmin on the full 1000-grid (t -> ~t/1000), so [1000,750,500,250]
    # ends at sigma 0.25. MUST stay identical in native_infer/eval_multi/long_rollout.
    denoising_steps = [int(s) for s in args.denoise_list.split(",")]
    log(f"denoising_steps ({len(denoising_steps)}-step, explicit) = {denoising_steps}")

    # resume: load the latest saved generator + critic dits (before prepare, into raw modules) + step
    resume_step = 0
    if args.resume and not args.dry_run:
        import json
        pj = os.path.join(run_dir, "progress.json")
        if os.path.exists(pj):
            resume_step = int(json.load(open(pj))["step"])
            args.student_ckpt = runlib.dit_path(run_dir, resume_step)
            critic_resume = os.path.join(run_dir, "critic", f"critic_step{resume_step}.safetensors")
            ema_resume = os.path.join(run_dir, "ema", f"ema_step{resume_step}.safetensors")
            log(f"RESUME from step {resume_step}: student<-{os.path.basename(args.student_ckpt)} critic<-{os.path.basename(critic_resume)} ema<-{os.path.basename(ema_resume)}")
        else:
            log(f"--resume set but no progress.json in {run_dir}; starting fresh")
            args.resume = None

    if not args.dry_run:
        if resume_step:
            load_dit_ckpt(student, args.student_ckpt)
            load_dit_ckpt(critic, critic_resume)
            if ema is not None:
                load_dit_ckpt(ema, ema_resume)
        else:
            log(f"loading: student<-ODE-init, critic<-teacher{', ema<-student-init' if ema is not None else ' (no ema)'} ...")
            load_dit_ckpt(student, args.student_ckpt)
            load_dit_ckpt(critic, args.teacher_ckpt)
            if ema is not None:
                load_dit_ckpt(ema, args.student_ckpt)    # EMA starts == student init
        load_dit_ckpt(teacher, args.teacher_ckpt)
        all_files = sorted(glob.glob(os.path.join(args.data, "clip_*.pt")))
        train_files = runlib.split_clip_files(all_files, "train", args.val_frac)
        train_files = runlib.oversample_clip_files(train_files, args.balance, args.balance_alpha,
                                                   args.balance_max_repeat, log=log)
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
            args.max_steps, args.save_every, args.warmup, args.log_every, args.critic_steps = 12, 6, 2, 2, 2
            args.ema_start = 3
        log("DRY RUN: tiny model, synthetic data")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate_precomputed, drop_last=True, pin_memory=True)

    def _adamw(params, lr):
        # 8-bit Adam (bitsandbytes); is_paged offloads opt state to CPU. NB: the DMD step's ~66GB
        # peak is ACTIVATIONS (rollout graph + teacher CFG + critic scoring live together), NOT
        # the optimizer, so paging didn't move the peak — kept is_paged off (faster). The real
        # footprint lever is the rollout/scoring activations (TODO: checkpoint rollout / drop CFG).
        # ★ FIX: reference uses betas=(0.0, 0.999) + weight_decay 0.01. beta1=0 (no momentum)
        # is standard for the non-stationary GAN-like fake-score objective — momentum (0.9)
        # made the critic lag the moving generator, compounding the 1:1 starvation.
        if args.opt8bit and not args.dry_run:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01, betas=(0.0, 0.999),
                                       is_paged=bool(args.paged_opt))
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, betas=(0.0, 0.999))
    opt_g = _adamw(student.parameters(), args.lr)
    opt_c = _adamw(critic.parameters(), args.lr_critic)

    def lr_lambda(s):
        return (s + 1) / args.warmup if s < args.warmup else 1.0
    slr_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    slr_c = torch.optim.lr_scheduler.LambdaLR(opt_c, lr_lambda)

    class _Counter:
        def __init__(self): self.step = 0
        def state_dict(self): return {"step": self.step}
        def load_state_dict(self, sd): self.step = int(sd["step"])
    counter = _Counter()

    # Teacher is frozen + only used in no_grad forward, but we FSDP-shard it too (10GB replicated
    # -> ~2.5GB/card) so 3 5B models fit. Freeze BEFORE prepare so FSDP wraps it without grad hooks.
    teacher.requires_grad_(False).eval()
    if ema is not None:
        ema.requires_grad_(False).eval()                   # frozen deploy shadow (FSDP-sharded like student)
        student, critic, teacher, ema, opt_g, opt_c, dl, slr_g, slr_c = accel.prepare(
            student, critic, teacher, ema, opt_g, opt_c, dl, slr_g, slr_c)
    else:
        student, critic, teacher, opt_g, opt_c, dl, slr_g, slr_c = accel.prepare(
            student, critic, teacher, opt_g, opt_c, dl, slr_g, slr_c)
    neg = neg.to(accel.device, dtype=torch.bfloat16)
    sched = FlowMatchScheduler(shift=5.0).to(accel.device)
    ac = lambda: torch.autocast(device_type=accel.device.type, dtype=torch.bfloat16)

    # ── frozen Wan2.2 VAE for the rollout RE-ANCHOR (CF base.py:154-167). Built ONCE here, after
    #    accel.prepare, on each rank's device (~2.8GB, no_grad). Skipped under --dry_run (tiny
    #    synthetic latents have no real VAE manifold) and when re-anchor is disabled
    #    (reanchor_every<=0 -> OLD no-re-anchor behavior, no VAE cost / no ckpt needed). ──
    vae = None
    if not args.dry_run and args.reanchor_every and args.reanchor_every > 0:
        sys.path.insert(0, os.environ.get("GM_WAN_DIR", os.path.join(GMroot, "vendor_wan")))
        from modules.vae2_2 import Wan2_2_VAE
        _wan = os.environ.get("GM_WAN_CKPT", "/opt/dlami/nvme/zhiyangdeng/_shared/base_models/wan2.2-ti2v-5b-dit")
        vae = Wan2_2_VAE(vae_pth=os.path.join(_wan, "Wan2.2_VAE.pth"), device=accel.device)   # frozen, ~2.8GB
        log(f"VAE re-anchor ON: R={args.reanchor_every} W_dec={args.reanchor_win} "
            f"(decode {4 * args.reanchor_win - 3} px frames, keep last) | Wan2.2 VAE @ {_wan}")

    counter.step = resume_step                                 # dits already loaded pre-prepare
    step = counter.step
    _ls, _lt = step, time.time()
    last_gl = torch.tensor(float("nan"), device=accel.device)   # persist last generator loss across non-gen iters
    last_gng = float("nan")

    def _clip(model, max_norm):
        # accel.clip_grad_norm_ breaks with >1 FSDP model (its matcher does a tensor `==`).
        # FSDP exposes its own clip_grad_norm_ (handles sharded grads); bare module -> torch util.
        if hasattr(model, "clip_grad_norm_"):
            return model.clip_grad_norm_(max_norm)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    student.train(); critic.train()
    data_iter = iter(dl)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            ds.set_epoch(getattr(ds, "epoch", 0) + 1); data_iter = iter(dl); batch = next(data_iter)
        latent = batch["latent"].to(accel.device)              # [B,48,F,Hl,Wl]
        context = batch["context"].to(accel.device)            # [B,F,512,4096]
        B, _, Fl = latent.shape[:3]
        # ── rollout length N (frames beyond frame0). Decoupled from clip length Fl so L=1+N>F
        #    exercises the windowed eviction (fires at frame_index > sink+kv_window). RANK-SYNCED
        #    (FSDP all-gather shapes must match every step; CF base.py:140). BACKWARD-COMPATIBLE:
        #    --num_gen_frames set => fixed N (NO LONGER capped at Fl-1). ──
        if args.num_gen_frames is not None:
            N = args.num_gen_frames
        else:
            _nt = torch.randint(args.rollout_min, args.rollout_max + 1, (1,), device=accel.device)
            if accel.num_processes > 1:
                torch.distributed.broadcast(_nt, src=0)        # identical L across ranks (CF base.py:140)
            N = int(_nt.item())
        initial = latent[:, :, 0:1]                            # real clean start
        # per-frame prompts for frames 0..N. context has only Fl entries; for 1+N>Fl HOLD/tile the
        # last authored per-frame prompt (context[:, Fl-1]) for frames >=Fl (CF single-conditional
        # hold, rolling_forcing_training.py:178-184). .expand is a view; cat materializes contiguous.
        if 1 + N <= Fl:
            prompts = context[:, :1 + N]                       # full-clip per-frame prompts (frame k uses [:,k])
        else:
            last = context[:, Fl - 1:Fl]                       # [B,1,L,4096] final authored per-frame prompt
            pad = last.expand(B, (1 + N) - Fl, *context.shape[2:])
            prompts = torch.cat([context, pad], dim=1)         # [B, 1+N, L, 4096] (held last prompt for >=Fl)
        assert prompts.shape[1] == 1 + N, f"prompt tile failed: {prompts.shape[1]} != {1 + N}"
        uncond = neg.view(1, 1, *neg.shape).expand(B, 1 + N, *neg.shape)

        # ONE shared grad rollout per step, used for BOTH the generator (grad) and the critic
        # (detached). CRITICAL (FSDP, verified in test_fsdp_rollout_grad.py): a student rollout
        # that is NOT immediately followed by a student backward (a no_grad rollout, OR a grad
        # rollout whose gen-update is skipped) zeroes the student grad on the NEXT gen step (FSDP
        # backward setup is consumed). So the generator updates EVERY step, the student backward
        # runs BEFORE the critic backward, and the critic reuses the detached samples (critic_steps
        # inner updates give it a lead without a 2nd rollout). teacher/critic no_grad forwards in
        # the dmd loss are separate FSDP modules and do NOT interfere (verified).
        _prof = args.profile_mem and step < 3 and accel.is_main_process
        def _mem(tag):
            if _prof:
                torch.cuda.synchronize()
                log(f"  [mem step{step}] {tag}: peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB "
                    f"cur {torch.cuda.memory_allocated()/2**30:.1f}GB")
        if args.profile_mem:
            torch.cuda.reset_peak_memory_stats()
        # last-K scoring: pin grad window to the tail iff windowed + rollout exceeds K (else None =
        # OLD random window; small-L Keff<=N handled below). dmd_score_frames<=0 disables (back-compat).
        Keff = min(args.dmd_score_frames, N) if args.dmd_score_frames and args.dmd_score_frames > 0 else None
        with ac():
            gen, grad_mask = self_forcing_rollout(student, initial, prompts, N, denoising_steps,
                                                  args.grad_horizon, sched,
                                                  kv_window=args.kv_window, rope_cap=args.rope_cap,
                                                  sink_size=args.sink_size, score_last_k=Keff,
                                                  vae=vae, reanchor_every=args.reanchor_every,
                                                  reanchor_win=args.reanchor_win)
        _mem("after rollout")

        # ── scored-tail slicing (CF base.py:154-167): score only frame0 + the last Keff generated
        #    frames. The FULL rollout `gen` still ran end-to-end (so eviction was exercised); only
        #    the LOSS inputs are the tail. frame0 stays (the gate requires a clean t=0 anchor at
        #    index 0). When Keff is None or Keff>=N the tail == full rollout (no-op). ──
        if Keff is not None and Keff < N:
            tail_sl = torch.cat([gen[:, :, :1], gen[:, :, 1 + N - Keff:]], dim=2)         # [B,C,1+Keff,H,W]
            prompts_t = torch.cat([prompts[:, :1], prompts[:, 1 + N - Keff:]], dim=1)     # [B,1+Keff,L,4096]
            uncond_t = uncond[:, :1 + Keff]                                                # auto-sized neg
            grad_mask_t = torch.cat([grad_mask[:1], grad_mask[1 + N - Keff:]])            # [1+Keff] bool
        else:
            tail_sl, prompts_t, uncond_t, grad_mask_t = gen, prompts, uncond, grad_mask

        # ── generator update FIRST (student backward before critic), every step ──
        with ac():
            gl, glog = dmd_generator_loss(teacher, critic, tail_sl, prompts_t, uncond_t, grad_mask_t,
                                          sched, args.real_guidance_scale)
        _mem("after dmd_loss")
        accel.backward(gl)
        _mem("after gen backward")
        gng = _clip(student, 10.0)                              # ref max_grad_norm_generator=10.0
        opt_g.step(); opt_g.zero_grad(); slr_g.step()
        # generator EMA: hard-copy until ema_start, then decay (updated every generator step)
        if ema is not None:
            ema_update(ema, student, decay=(0.0 if step < args.ema_start else args.ema_decay))
        last_gl = gl.detach(); last_gng = float(gng)
        _mem("after opt_g.step")

        # ── critic update(s): flow-denoise the student's own samples (detached), critic_steps times.
        #    Use the SAME scored tail (detached) + tiled prompts as the generator loss. ──
        gen_det = tail_sl.detach()
        for _ci in range(args.critic_steps):
            with ac():
                cl, clog = critic_loss(critic, gen_det, prompts_t, sched)
            accel.backward(cl)
            gnc = _clip(critic, 10.0)                           # ref max_grad_norm_critic=10.0
            opt_c.step(); opt_c.zero_grad()
        _mem("after critic")
        slr_c.step()

        step += 1; counter.step = step
        if step % args.log_every == 0:
            avg_c = accel.gather(cl.detach()).mean().item()
            avg_g = accel.gather(last_gl).nanmean().item()
            gng = last_gng
            now = time.time(); sps = (step - _ls) / max(now - _lt, 1e-6); _ls, _lt = step, now
            log(f"step {step:6d}/{args.max_steps}  dmd {avg_g:+.4f}  critic {avg_c:.4f}  "
                f"gnorm_g {float(gng) if gng is not None else float('nan'):.2f} gnorm_c {float(gnc):.2f}  {sps:.2f} it/s")
            if _jsonl is not None:
                _jsonl.log({"step": step, "dmd": avg_g, "critic": avg_c, "it_per_s": sps})
        if step % args.save_every == 0 or step == args.max_steps:
            # NB: NOT accel.save_state — FSDP.optim_state_dict can't consolidate bitsandbytes 8-bit
            # Adam states (size mismatch). We save the consolidated DITS (params consolidate fine):
            # student=generator (deploy) + critic (for clean resume). Optimizers restart on resume
            # (fine for distillation). get_state_dict is COLLECTIVE — every rank calls it.
            accel.wait_for_everyone()
            sd_g = accel.get_state_dict(student)
            sd_c = accel.get_state_dict(critic)
            sd_e = accel.get_state_dict(ema) if ema is not None else None   # collective — every rank calls it
            if is_main:
                import json
                import safetensors.torch as st
                dpath = runlib.dit_path(run_dir, step)
                cpath = os.path.join(run_dir, "critic", f"critic_step{step}.safetensors")
                os.makedirs(os.path.dirname(dpath), exist_ok=True)
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                st.save_file({k: v.contiguous() for k, v in sd_g.items()}, dpath)
                st.save_file({k: v.contiguous() for k, v in sd_c.items()}, cpath)
                if sd_e is not None:
                    epath = os.path.join(run_dir, "ema", f"ema_step{step}.safetensors")   # DEPLOY weights
                    os.makedirs(os.path.dirname(epath), exist_ok=True)
                    st.save_file({k: v.contiguous() for k, v in sd_e.items()}, epath)
                with open(os.path.join(run_dir, "progress.json"), "w") as f:
                    json.dump({"step": step}, f)
                # keep only the last keep_last_states critic shards (dits kept by user)
                crs = sorted(glob.glob(os.path.join(run_dir, "critic", "critic_step*.safetensors")),
                             key=lambda p: int(p.split("step")[-1].split(".")[0]))
                for old in crs[:-args.keep_last_states]:
                    os.remove(old)
            accel.wait_for_everyone()
            log(f"saved step {step}: generator + critic dits + progress.json")
    if _jsonl is not None:
        _jsonl.close()
    log("DMD DONE")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        main()
