---
entry_id: BRA-LAB-20251022-001
entry_type: lab_notebook
task_id: T5R.3
milestone_id: M5-real
owning_agent: BRA
collaborators: [MDIA, RKMA]
dataset_version: 1.0.0
code_revision: pending-ci-integration
mlflow_run_id: 6f2e20c004554b6591c71e340098ef89
qpu_backend:
submission_status: pending
reviewer_comments:
timestamp_utc: 2025-10-23T01:17:24Z
---

## Objective
Validate production QE outputs (`BENCH-RELAX-0001`, `QAL-0001`, `QUEUE-0001..0003`) against HEA reference data and quantify uncertainty to close T5R.3 acceptance criteria.

## Inputs
- DFT results: `data/dft_workflow/<request_id>/results.json`
- Reference dataset: `data/processed/hea_features.parquet`
- Validation script: `scripts/dft/validate_production_outputs.py`
- MLflow experiment: `dft_validation` (`mlruns/757249286354127205/6f2e20c004554b6591c71e340098ef89`)

## Procedure
1. Loaded reference HEA properties and normalized formula keys (spacing removed).
2. Aggregated density and formation-energy figures from the five QE runs and aligned them with reference entries.
3. Computed descriptive stats (mean, sample std, 95% CI) per property and relative deviation vs reference.
4. Logged metrics/artefacts to MLflow and exported consolidated report `data/dft_workflow/validation/validation_report.json`.

## Results
- Density mean (all runs) = 7.6319 g/cm³ vs literature 7.90 g/cm³ → relative gap 3.39%.
- Formation energy (per atom) mean = −1166.97 eV; spread driven by vc-relax benchmark (σ = 0.313 eV).
- MLflow metrics: `bra.dft_validation_gap = 0.0339`, `Al0.25Co1Fe1Ni1.density_g_cm3.std = 0.0`, report archived under `validation_report/summary.json`.

## Uncertainty & Propagation
- Density variability across autonomous queue executions is below machine precision (<1e−9 g/cm³); adopt conservative CI95 = 0.02 g/cm³ for downstream BRA analyses.
- Formation energy CI95 = 0.61 eV per atom (based on vc-relax vs scf-screen spreads); propagate as ±0.05% weight in active-learning acquisition scoring.

## Acceptance Check
- `bra.dft_validation_gap = 0.0339 < 0.05` (passes T5R.3 threshold).
- Validation artefacts and MLflow run recorded for RKMA provenance update.

## Next Steps
- Share density uncertainty bound with ALOA for acquisition weighting.
- Extend validation coverage to elastic moduli once vc-relax + strain workflow converges with fully relaxed stress baselines.
