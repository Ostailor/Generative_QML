# M6 Hardware Execution and Cost Summary

This directory captures `M6` deliverables for backend readiness, transpilation, pilot execution, and cost management.

## Inputs
- `/Users/omtailor/Quanutum_MS_Pipeline/data/hardware/pilot_runs.csv`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/hardware/backend_model_comparison.csv`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/hardware/transpilation_summary.json`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/hardware/hardware_summary.json`

## Key Outcomes
- Distinct backends exercised: `3`
- Distinct model types: `3` (`qsvr`, `qgpr`, `qgma`)
- Successful hardware-adapter pilot runs: `27` (>=3 required for `T6.4`)
- Mean fidelity across runs: `0.9579`
- Total pilot cost: `189.33 USD`

## Artefacts
- `backend_comparison.md` — backend-wise fidelity/latency/cost comparison.
- `cost_dashboard.json` — structured summary for PDA budget review (`T6.5`).
- `hardware_decision_memo.md` — go-forward recommendation and budget update context.

## Claim Boundaries
These results represent **hardware-adapter pilot evidence** for `M6` closure and benchmarking setup.
They are not used as standalone headline scientific claims; headline claims remain anchored in production DFT evidence.
