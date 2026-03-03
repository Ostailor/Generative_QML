# Real DFT Campaign Release v1

This package closes `T5R.5` by freezing the production campaign artefacts for `t5r4-20260211-fasttrack-221-mw4`.

## Contents
- `campaign/` — Full campaign snapshot copied from `data/dft_workflow/campaigns/t5r4-20260211-fasttrack-221-mw4`.
- `release_info.json` — High-level release metadata and headline real-only metrics.
- `release_manifest.json` — SHA256 + size inventory for reproducibility.
- `reproduction_runbook.md` — Deterministic replay and validation instructions.

## Evidence Policy
This release is **real-backed**: headline claims are derived from production DFT and queue outputs.
Simulated/mock artefacts are excluded from primary claims.
