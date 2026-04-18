# Real DFT Campaign Release v1

This package closes `T5R.5` by freezing the production campaign artefacts for `top-tier-nonqpu-gpu-20260303T195828Z-vc-20260303t195829z`.

## Contents
- `campaign/` — Full campaign snapshot copied from `data/dft_workflow/campaigns/top-tier-nonqpu-gpu-20260303T195828Z-vc-20260303t195829z`.
- `release_info.json` — High-level release metadata and headline real-only metrics.
- `release_manifest.json` — SHA256 + size inventory for reproducibility.
- `reproduction_runbook.md` — Deterministic replay and validation instructions.

## Evidence Policy
This release is **real-backed**: headline claims are derived from production DFT and queue outputs.
Simulated/mock artefacts are excluded from primary claims.
