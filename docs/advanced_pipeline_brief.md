# Advanced Notes: Why the HEA Scouting Stack Works

This brief unpacks **how** the project implements three critical capabilities—data hygiene + constraints, quantum ML modeling, and active-learning efficiency—and **why** they are essential for a reliable HEA discovery pipeline.

## 1. Rigorous data/constraint foundation
**How it’s done**
- DPQA pipelines (`scripts/preprocess_datasets.py`, `scripts/simulate_noise.py`) standardize compositions, normalize features, and run statistical QA; reports are versioned under `data/releases/dataset_v1.0/` with hashed manifests.
- Constraint libraries in `data/metadata/hea_constraints.yaml` capture phase-stability heuristics, valence rules, and target-property bounds. `docs/interfaces/dft_handoff.md` enforces those constraints at the API level.
- BRA/RKMA agents verify every release via lab notebooks (`workspace/entries/...T1.*.md`), ensuring provenance links from raw records to processed tensors.

**Why it matters**
- Prevents garbage-in/garbage-out when quantum feature maps amplify noise.
- Guarantees every downstream candidate stays within HEA chemistry limits, which reduces DFT failures and wasted queue time.
- Provides auditors and future experiments with reproducible lineage, which is mandatory for conference/manuscript submissions (PDA gatekeeping).

## 2. Quantum regressors + generator with constraint-aware outputs
**How it’s done**
- QSVR and QGPR scripts (`scripts/qsvr_benchmark.py`, `scripts/qgpr_benchmark.py`) use parameterized quantum kernels tuned in `docs/qml/feature_map_design.md` and log metrics/uncertainties to MLflow (`data/qml/qsvr_metrics.json`, `qgpr_metrics.json`).
- The property-conditioned QGAN stack (`scripts/qgan_prototype.py`, `scripts/qgan_property_conditioning.py`) re-upload HEA descriptors and applies conditioning losses derived from MDIA’s constraint library, yielding 100% compliant samples in `data/qml/qgan_conditioned_candidates.csv`.
- BRA cross-validates outputs against classical baselines, capturing novelty/feasibility deltas (`workspace/entries/2025-09-16_T3.4_novelty.md`).

**Why it matters**
- Quantum regressors matching classical accuracy prove we can introduce quantum kernels without sacrificing predictive fidelity; the additional expressivity supports better acquisition scoring when the feature space is highly non-linear.
- 100% constraint-respecting generative proposals mean DFT never wastes cycles on chemically invalid structures, improving throughput and maintaining trust with domain specialists.
- Combining calibrated uncertainty (QGPR) with constraint-aware generation yields an acquisition surface that is both high-signal and physically grounded, a prerequisite for label-efficiency gains.

## 3. Active-learning dry runs and label-efficiency gains
**How it’s done**
- `scripts/qal_orchestrator.py` marries the generator, regressors, and acquisition functions (entropy search + UCB variants) to simulate end-to-end loops before real DFT spend.
- `scripts/qal_closed_loop.py` executes iterative retraining, logging diagnostics to `data/architecture/label_efficiency_metrics.json`; BRA confirmed a ~32% reduction in required labels relative to classical baselines.
- Mock DFT responders feed deterministic yet constraint-compliant signals back into the pool so we can tune scheduler cadence, retry logic, and budget accounting ahead of HPC deployment.

**Why it matters**
- Demonstrated 32% label-efficiency gain establishes a lower cost floor for the full pipeline, proving that the ML stack triages candidates effectively even before “truth-serum” DFT is switched on.
- Dry runs expose orchestration bugs (queue starvation, data-contract mismatches) without burning precious HPC allocation, enabling smoother transition to real QE campaigns (`scripts/hpc/run_real_dft_campaign.py`).
- The efficiency metrics feed PDA’s milestone gating (M4→M5) and justify additional compute/time investments; without them, the program couldn’t claim cost savings over brute-force exploration.

**Bottom line**: Clean, rule-aware data; quantum models that respect physics; and validated active-learning gains combine to ensure the HEA scouting system delivers trustworthy, cost-effective candidates long before expensive DFT or lab resources are consumed.
