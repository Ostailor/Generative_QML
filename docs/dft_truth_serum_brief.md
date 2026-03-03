# Advanced Notes: How “DFT Truth Serum” Fits the Loop

This explainer covers **how** we integrate Density Functional Theory (DFT) into the scouting pipeline and **why** the supporting infrastructure keeps accuracy high while limiting expensive Quantum ESPRESSO (QE) runs.

## 1. Treat DFT as the ground-truth oracle
**How it’s done**
- MDIA manages Quantum ESPRESSO inputs/outputs via the handoff spec (`docs/interfaces/dft_handoff.md`). Each ML-selected alloy gets a reproducible package with `structure.cif`, `metadata.json`, and QE parameters.
- `scripts/dft/run_dft_workflow.py` executes production Quantum ESPRESSO jobs locally/HPC, validates metadata contracts, and logs results to MLflow for campaign-scale operation.
- `docs/dft/dft_run_log.md` and lab notebooks (e.g., `workspace/entries/2025-10-22_T5R3_dft_validation.md`) capture per-request energies, densities, and convergence flags, ensuring physics-grade fidelity is traceable.

**Why it matters**
- DFT outputs anchor the entire program: regressors, acquisition rules, and generative targets are calibrated against these “truth” values, preventing drift from purely statistical guesses.
- Without physics-level validation, quantum ML claims (better kernels, generators) would have no benchmark, undermining manuscript-ready evidence.

## 2. Use ML scoring to protect the QE budget
**How it’s done**
- The active-learning orchestrator (`scripts/qal_orchestrator.py`) ranks candidates using QSVR/QGPR predictions plus uncertainty estimates; only top-ranked alloys pass to the DFT queue.
- `scripts/hpc/run_real_dft_campaign.py` batches those leads, writes QE-ready folders under `data/dft_handoff/input/`, and launches jobs through `run_queue` with throttles (`--max-workers`, retry budgets) so cost stays predictable.
- BRA monitors queue summaries (`data/dft_workflow/queue_runs/`) and MLflow metrics (`mdia.dft_jobs_completed`, `aloa.real_label_efficiency_gain`) to ensure we are not over-spending per iteration.

**Why it matters**
- QE runtime scales with system size; blindly simulating every candidate would exhaust HPC allocation before we gathered useful statistics.
- Routing only the “best leads” keeps the signal-to-cost ratio high: every DFT label is informative for retraining, so the loop stays sample efficient.

## 3. Close the loop to keep accuracy high and cost low
**How it’s done**
- After each QE batch, `scripts/dft/validate_production_outputs.py` compares energies/densities against reference datasets, producing validation reports (`data/dft_workflow/validation/validation_report.json`).
- The new labels feed back into the training pool; `scripts/qsvr_benchmark.py` and `scripts/qgpr_benchmark.py` retrain with the fresh DFT data, while acquisition weights are updated to reflect improved certainty.
- PDA/ALOA track label-efficiency metrics (`data/architecture/label_efficiency_metrics.json`) to verify that accuracy gains keep pace with cost; the latest campaign delivered a 0.80 efficiency score vs the 0.30 gate.

**Why it matters**
- Continuous feedback ensures the ML stack stays calibrated to real physics, preventing divergence between “cheap” predictions and “expensive” truth.
- Demonstrating high accuracy per DFT dollar is central to the project’s value proposition (cost-efficient discovery). Without the loop, we’d either overspend on simulations or under-deliver on predictive quality.

**Bottom line**: DFT provides authoritative labels, but the scouting loop’s quantum ML prioritization means we only pay for the simulations that truly sharpen the models—keeping accuracy high while the simulation bill stays under control.

## Appendix: Interpreting the “Real DFT Progress” image
- **What it shows**: The chart in `docs/slides/assets/dft_kpis_clean.png` compares two simple metrics—label-efficiency gain and number of valid candidates—against their acceptance targets (0.30 and 10, respectively). Bars colored blue represent what the latest Quantum ESPRESSO campaign actually achieved, while gray bars show the minimum thresholds set in `TASKS.md` for T5R.4 sign-off.
- **Why it matters**: Hitting 0.80 label efficiency and 24 valid candidates demonstrates that the closed-loop prioritization is working exactly as intended: every DFT label pulled from the HPC queue carries outsized learning value, so the program saves simulation budget without sacrificing accuracy. It also confirms the gate criteria PDA established for moving into manuscript and benchmarking workstreams, giving stakeholders a quick visual that the expensive physics runs are returning the promised value.
