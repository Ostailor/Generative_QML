# Hardware Decision Memo (`T6.5`)

## Decision
Approve progression to `M7` benchmarking using the current hardware-adapter pilot stack and updated cost controls.

## Evidence
- Pilot runs: `27` successful runs across `3` backends and `3` model types.
- Aggregate cost: `189.33 USD` for the pilot envelope.
- Mean fidelity: `0.9579`.
- Reference data:
  - `/Users/omtailor/Quanutum_MS_Pipeline/data/hardware/hardware_summary.json`
  - `/Users/omtailor/Quanutum_MS_Pipeline/docs/hardware/cost_dashboard.json`

## Budget Update
1. Set primary sweep backend to `ibm_perth_adapter` for cost efficiency.
2. Reserve `quantinuum_h1_adapter` for high-fidelity confirmation sweeps only.
3. Maintain monthly checkpoint on cost/fidelity drift before each benchmark cycle.

## Status
- `pda.hardware_budget_updated = 1`
- Milestone transition: `M6 -> M7` approved.
