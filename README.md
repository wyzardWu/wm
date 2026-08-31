# wm — interactive world model experiments

One month of work across three lines (Aug 2026):

- **vrisingroam/** — V Rising roaming WM on ReactiveGWM (Wan2.2-TI2V-5B + per-block additive
  action injection). Data pipeline (timeline/actions/filters/streaming cuts), 8-GPU cache
  precompute, bidirectional training. `ReactiveGWM/` is the upstream fork incl. the
  unreleased DiffSynth fork (see PROVENANCE.txt) with our `vrising` profile.
  `cf_distill_3stage/` — causalize → consistency-distill → DMD stack (used for EYBX too).
- **eybxroam/** — No Rest for the Wicked scene-switching model: per-frame CA prompt with
  [action16 ⊕ scene16] context; screen-space action labels; stage1 causalize + CD to a
  2-step student; `play_server.py` = real-time browser demo (WASD + scene keys over WS).
- **ltxwm/** — LTX-2.3 22B action injection (ABot 8k clips): frame-fold per-frame text
  cross-attention ([caption1024 | move32 | view32] per latent frame), v3.1 decorrelated
  action vocabulary (centered-cosine surgery), LoRA rank32. `ltxwm/` is our code
  (strategy/patch/probe/AdaLN-ablation), `LTX-2/` upstream trainer (+ final-checkpoint
  self-delete bugfix), `alayaworld/` upstream stack (+ FA3-hardcode fix, chunk-script
  patches, our fold port in ltxwm_port/). WildWorld (MH Wilds) pilot prep: slicing,
  VLM action-sentence tables, three-slot strategy ([caption|player|boss|cam]).

Data, weights, latents and outputs are not in this repo (paths reference the cluster).
