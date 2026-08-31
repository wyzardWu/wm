# Archived configs

Superseded / one-off configs kept for provenance. Not on any current run path.
Archived 2026-06-06. Canonical configs live one level up in `configs/` and target the
**sf3_casual_forcing_2** lineage (the corrected/aligned re-train).

| File | Origin | Why archived |
|---|---|---|
| `stage1_ar_8card.yaml` | original 8-card Stage 1 AR-TF, old `sf3_casual_forcing/` lineage (pre-alignment semantics) | superseded by canonical `stage1_ar.yaml` (CF++-aligned, `_2` lineage). |
| `stage1_ar_4card.yaml` | 4-card variant of the old 8-card config, old lineage | same old (pre-alignment) recipe; superseded by canonical `stage1_ar.yaml`. |
| `stage2_cd_8card.yaml` | original 8-card Stage 2 CD, old lineage | superseded by canonical `stage2_cd.yaml` (`_2` lineage, grad_accum=1). |
| `stage2_cd_4card.yaml` | 4-card variant, old `sf3_casual_forcing/` output path | superseded by canonical `stage2_cd.yaml`. |
| `stage3_dmd_26win_resume5400.yaml` | one-off resume of Stage 3 from `state-5400`, long-rollout disabled | 26-window baseline ablation; canonical is `stage3_dmd.yaml`. |
