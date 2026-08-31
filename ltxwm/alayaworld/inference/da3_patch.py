"""Robust scale-only recovery for DA3 camera-conditioned depth on (near-)colinear
camera trajectories — applied at the inference layer, no engine/DA3 source edits.

Why: DA3 aligns its predicted depth to the input camera poses with a full Sim(3)
Umeyama (rotation + translation + scale) over each short window, in
`depth_anything_3.api._align_to_input_extrinsics_intrinsics` ->
`depth_anything_3.utils.pose_align.align_poses_umeyama`, then rescales the depth by
that scale (`prediction.depth /= scale`). Over a COLINEAR window — a straight
forward dolly, a pure sideways/vertical move, i.e. any constant-direction segment —
the camera centers are rank-1, so evo's Umeyama raises
`GeometryException("Degenerate covariance rank, Umeyama alignment is not possible")`.
The engine then drops camera conditioning entirely and the depth stays in DA3's
arbitrary monocular scale, so forward motion renders as a uniform ZOOM instead of
real parallax.

Fix: the ROTATION is what needs rank>=2; the SCALE is a scalar (the ratio of the
two trajectories' extents) and is well-defined even for a colinear/straight path.
This wraps `align_poses_umeyama` so that, on exactly that degeneracy, it returns a
rotation-free scale-only result (R=I, t=0, s = extent(ref centers)/extent(est
centers)). `prediction.depth /= s` then applies the correct metric scale — forward
motion becomes real parallax with no change to the user's pose (so no camera-
following-benchmark penalty). Non-degenerate windows are untouched (original path).

Call `apply_da3_robust_scale()` once after the engine has loaded DA3 (so
`depth_anything_3` is importable); it is idempotent and safe to call repeatedly.
"""
from __future__ import annotations


def _robust_scale(ext_ref, ext_est) -> float:
    """Umeyama scale (est -> ref) recovered from camera-center extents only.
    ext_* are world-to-camera [N,3or4,4]; centers are the c2w translations."""
    import numpy as np
    from depth_anything_3.utils.geometry import affine_inverse_np

    def _centers(ext):
        c2w = np.asarray(affine_inverse_np(np.asarray(ext)))
        return c2w[:, :3, 3]

    cref, cest = _centers(ext_ref), _centers(ext_est)
    sref = float(np.sqrt(((cref - cref.mean(0)) ** 2).sum(1).mean()))
    sest = float(np.sqrt(((cest - cest.mean(0)) ** 2).sum(1).mean()))
    s = max(sref, 1e-8) / max(sest, 1e-8)
    # guard the pathological stationary-camera degeneracy (both extents ~0)
    return s if (np.isfinite(s) and s > 0.0) else 1.0


def apply_da3_robust_scale() -> bool:
    """Monkeypatch align_poses_umeyama to fall back to scale-only on Sim(3)
    degeneracy. Returns True if applied (DA3 importable), False otherwise."""
    try:
        import depth_anything_3.api as _api
        from depth_anything_3.utils import pose_align as _pa
    except Exception:
        return False  # DA3 not importable yet — call again after the engine loads it

    import numpy as np

    orig = _pa.align_poses_umeyama
    if getattr(orig, "_robust_scale_wrapped", False):
        return True

    def wrapped(ext_ref, ext_est, return_aligned=False, **kwargs):
        try:
            return orig(ext_ref, ext_est, return_aligned=return_aligned, **kwargs)
        except Exception as exc:  # evo GeometryException et al.
            msg = str(exc).lower()
            if not ("degenerate" in msg or "covariance" in msg or "umeyama" in msg or "rank" in msg):
                raise  # unrelated failure — don't mask it
            s = _robust_scale(ext_ref, ext_est)
            r = np.eye(3, dtype=np.float64)
            t = np.zeros(3, dtype=np.float64)
            print(
                f"[da3_patch] Umeyama degenerate (colinear window) -> scale-only s={s:.4f}",
                flush=True,
            )
            if return_aligned:
                # aligned extrinsics are only consumed when align_to_input_ext_scale=False,
                # which is not our path (we use the scale). Return input as a safe placeholder.
                return r, t, s, np.asarray(ext_est)
            return r, t, s

    wrapped._robust_scale_wrapped = True
    # Patch both the definition module and api's already-bound reference.
    _pa.align_poses_umeyama = wrapped
    _api.align_poses_umeyama = wrapped
    return True
