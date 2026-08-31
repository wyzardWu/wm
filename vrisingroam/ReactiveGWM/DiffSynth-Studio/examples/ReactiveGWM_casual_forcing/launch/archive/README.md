# Archived launchers

One-off launch/watch scripts hard-bound to a specific past run. Kept for provenance.
Archived 2026-06-06. Canonical launchers live one level up in `launch/`.

| File | Origin | Why archived |
|---|---|---|
| `resume_stage3_4card.sh` | ad-hoc 8→4 card FSDP reshard resume from `state-4800` (2026-05-24) | hard-codes a specific state dir + GPU 0-3; one-time recovery, not reusable. |
| `watch_stage1_aligned.sh` | background watcher for the `cf_stage1_aligned` tmux run | hard-codes a specific log path + tmux session name; bound to that run. |
