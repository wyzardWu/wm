"""
Self-contained flow-matching (rectified-flow) scheduler for Wan2.2.

Math copied verbatim from DiffSynth-Studio's FlowMatchScheduler "Wan" template
(`DiffSynth-Studio/diffsynth/diffusion/flow_match.py`) and confirmed identical to
Incantation's `utils/scheduler.py` convention:

    x_t      = (1 - sigma) * x0 + sigma * noise        (add_noise)
    v_target = noise - x0                              (training_target)
    loss     = MSE(v_pred, v_target) * training_weight(t)

Wan defaults: shift=5, num_train_timesteps=1000, sigma in (0,1].
We index sigmas/timesteps directly (avoids the float-match argmin hazard the
DiffSynth code warns about).
"""

import math
import torch


class FlowMatchScheduler:
    def __init__(self, num_train_timesteps=1000, shift=5.0, denoising_strength=1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        sigmas = torch.linspace(denoising_strength, 0.0, num_train_timesteps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas                                  # [N], in (0, ~1]
        self.timesteps = sigmas * num_train_timesteps         # [N], ~[0,1000)
        self._build_weights()

    def _build_weights(self):
        # bell-shaped "bsmntw" per-timestep loss weight (DiffSynth set_training_weight)
        steps = self.num_train_timesteps
        t = self.timesteps
        y = torch.exp(-2 * ((t - steps / 2) / steps) ** 2)
        y_shift = y - y.min()
        self.weights = y_shift * (steps / y_shift.sum())      # [N]

    def sample_timestep_ids(self, n, device):
        return torch.randint(0, self.num_train_timesteps, (n,), device=device)

    def add_noise(self, x0, noise, idx):
        """x_t = (1-sigma)*x0 + sigma*noise. idx: [B] long indices into sigmas.
        sigma broadcast per-sample over [B,C,F,H,W]."""
        sigma = self.sigmas.to(x0.device, x0.dtype)[idx]
        sigma = sigma.view(-1, *([1] * (x0.dim() - 1)))
        return (1 - sigma) * x0 + sigma * noise

    def training_target(self, x0, noise, idx=None):
        return noise - x0

    def training_weight(self, idx):
        return self.weights.to(idx.device)[idx]

    # ──────────────── Diffusion-Forcing per-frame noising (AR-teacher v2) ────────────────
    # ADDITIVE: these methods are new; the training/inference methods above are untouched,
    # so cd_gm / dmd_gm / ode_init that import this scheduler are unaffected. Used by
    # distill/ar_diffusion_gm.py to give each frame an INDEPENDENT noise level (Diffusion
    # Forcing, arXiv 2407.01392) — the fix for the single-shared-t exposure-bias collapse.
    # Convention reminder: sigmas[0]≈1 (max noise), sigmas[N-1]≈0 (min noise); timesteps =
    # sigmas*1000. So "noisier" == higher sigma == LOWER index.

    def add_noise_pf(self, x0, noise, idx_pf):
        """Per-frame noising. idx_pf: [B,F] long indices into sigmas (one level per frame).
        x_t = (1-sigma)*x0 + sigma*noise, sigma broadcast as [B,1,F,1,1] over [B,C,F,H,W]."""
        B, Fl = idx_pf.shape
        sigma = self.sigmas.to(x0.device, x0.dtype)[idx_pf]      # [B,F]
        sigma = sigma.view(B, 1, Fl, 1, 1)
        return (1 - sigma) * x0 + sigma * noise

    def sample_pf_timestep_ids(self, B, Fl, device, ar_order_bias=0.5):
        """Independent per-frame timestep ids [B,Fl] for Diffusion Forcing. Each frame draws
        an independent uniform noise level; with ar_order_bias>0 LATER frames are biased
        toward HIGHER sigma (noisier), matching AR generation order (earlier frames are the
        committed-cleaner history). ar_order_bias=0 => pure independent per-frame noise.
        Returns long indices; higher sigma maps to LOWER index (sigmas descend with index)."""
        N = self.num_train_timesteps
        if not (ar_order_bias > 0 and Fl > 1):
            # REFERENCE-FAITHFUL default (ReactiveGWM ar_tf.py:199): EXACTLY-uniform independent
            # per-frame ids. (The round((1-u)*(N-1)) path below under-samples idx 0 and N-1 by 2x.)
            return torch.randint(0, N, (B, Fl), device=device)
        # ar_order_bias>0 — our (non-reference) v2 extension: bias LATER frames toward higher sigma
        u = torch.rand(B, Fl, device=device)                                  # independent per frame
        frac = (torch.arange(Fl, device=device).float() / (Fl - 1))           # [Fl] 0..1
        p = (1 - ar_order_bias) * u + ar_order_bias * frac.unsqueeze(0)        # higher p => noisier
        return ((1 - p) * (N - 1)).round().long().clamp(0, N - 1)             # [B,Fl]

    def sigma_to_idx(self, sigma):
        """Nearest sigma-grid index for given sigma value(s). sigma: [B] float tensor -> [B] long."""
        grid = self.sigmas.to(sigma.device)
        return (grid.unsqueeze(0) - sigma.reshape(-1, 1)).abs().argmin(dim=1)

    # ───────────────────────── inference (sampling) ─────────────────────────
    # The training methods above are unchanged; these add a rectified-flow Euler
    # sampler for rollout/eval, mirroring Incantation utils/scheduler.py and
    # DiffSynth FlowMatchScheduler.set_timesteps/step. We sample on a SUBSET of the
    # 1000-step train grid (num_inference_steps points).

    def set_timesteps(self, num_inference_steps, denoising_strength=1.0, device=None):
        """Build a descending inference sigma/timestep schedule.

        linspace(strength, 0, N+1)[:-1] -> N sigmas (the trailing exact-0 is dropped;
        the final Euler step targets sigma=0 explicitly in step()). Then the Wan shift
        transform sigma' = shift*sigma/(1+(shift-1)*sigma). timesteps = sigma'*1000.
        Stores self.infer_sigmas / self.infer_timesteps (both length N)."""
        sigmas = torch.linspace(denoising_strength, 0.0, num_inference_steps + 1)[:-1]
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if device is not None:
            sigmas = sigmas.to(device)
        self.infer_sigmas = sigmas                                   # [N], desc, >0
        self.infer_timesteps = sigmas * self.num_train_timesteps     # [N], ~(0,1000)
        self.num_inference_steps = num_inference_steps
        return self.infer_timesteps

    def step(self, v_pred, x, i):
        """One Euler step of rectified-flow sampling.

        x(sigma) = (1-sigma)*x0 + sigma*noise, with v = d x / d sigma = noise - x0.
        Moving sigma_i -> sigma_{i+1} (smaller): x <- x + (sigma_{i+1}-sigma_i)*v_pred.
        The final step (i = N-1) targets sigma_next = 0, i.e. the clean x0 estimate."""
        sig = self.infer_sigmas.to(x.device, x.dtype)
        sigma = sig[i]
        sigma_next = sig[i + 1] if i + 1 < sig.shape[0] else x.new_zeros(())
        return x + v_pred.to(x.dtype) * (sigma_next - sigma)

    def to(self, device):
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)
        self.weights = self.weights.to(device)
        return self
