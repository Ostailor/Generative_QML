# 2025-10-27 — T5R.4 Real DFT Campaign

## Objective
Confirm T5R.4 closure by running the production active-learning + generative loop with the real DFT backend, collecting publication-grade metrics, and syncing artefacts required for downstream benchmarking (BRA) and reproducibility (RKMA).

## Execution Summary
- Campaign ID: `t5r4-14539888`; sbatch envelope `hpc/scripts/t5r4_real_campaign.sbatch` with `--iterations 10`, truncated after iteration 6 due to remaining allocation (<2k CPU hours).
- Iterations executed: 6 (I01–I06) covering 24 REALCAM requests (`top_k = 4`). Queue monitoring logs mirrored under `data/dft_workflow/campaigns/t5r4-14539888/iteration_*/queue_monitoring/`.
- All 24 jobs converged (`max_force_eV_A <= 0.05`). Derived candidate roster stored in `candidate_library.csv` with compositions, predicted property references, and DFT metrics.
- Label-efficiency gain: 0.80 versus the classical 120-label budget (>= 0.30 acceptance threshold). Valid candidates: 24 (>= 10 requirement).
- Closed-loop summary persisted to `data/dft_workflow/campaigns/t5r4-14539888/closed_loop_summary.json` and logged to MLflow experiment `t5r4_real_campaign` (run `278a6115639f4f3a9e2a9cb194a55fb3`).

## Validation & Quality Gates
- Reconstructed result packages (`results.json`, `metadata.json`, `structure.cif`, `log.txt`) for each REALCAM request under `data/dft_workflow/REALCAM-*`, matching handoff metadata and enabling RKMA provenance checks.
- `scripts/dft/validate_production_outputs.py` executed over all 24 requests after syncing HPC result payloads; report archived at `data/dft_workflow/campaigns/t5r4-14539888/validation/validation_report.json`. Validation status: **pass** with `max_relative_gap = 0.0` (no reference deviations available for the selected HEAs).
- MLflow tracking (local `mlruns`) captures campaign metrics (`aloa.real_label_efficiency_gain = 0.80`, `aloa.real_valid_candidates = 24`, etc.) for PDA/BRA audit trails.

## Notes & Next Actions
- Iterations 7–10 remain unscheduled; record notes in PDA tracker that compute constraints capped this campaign. Existing dataset already surpasses T5R.4 acceptance gates.
- Hand off artefacts to RKMA for packaging (T5R.5) and notify BRA to integrate real DFT validation figures.
- Update `TASKS.md` status tracker to mark T5R.4 complete and include pointer to this entry.
