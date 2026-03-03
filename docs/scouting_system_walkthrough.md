# High-Entropy Alloy Scouting System — How It Works

This note explains, in plain language, how the project’s scouting workflow turns raw data into a shortlist of high-entropy alloy (HEA) candidates before we run costly lab experiments.

## 1. Build the knowledge base (why the system trusts its inputs)
- **Curate + clean datasets**: DPQA agents use `scripts/preprocess_datasets.py` to harmonize HEA, perovskite, and single-atom alloy datasets, logging QA metrics in `data/releases/dataset_v1.0/`.
- **Encode guardrails**: MDIA codifies domain rules (`data/metadata/hea_constraints.yaml`) so any suggestion must obey chemistry, phase stability, and property limits.
- **Track provenance**: Every dataset, script run, and MLflow experiment is registered in `workspace/entries/lab_notebooks/`, giving a paper trail for later audits.

## 2. Let quantum ML models learn structure → property links
- **Quantum Support Vector Regressor (QSVR)**: `scripts/qsvr_benchmark.py` trains quantum kernel models that learn how alloy descriptors map to densities, formation energies, etc. These match classical accuracy but expose the system to quantum feature spaces.
- **Quantum Gaussian Process Regressor (QGPR)**: `scripts/qgpr_benchmark.py` adds calibrated uncertainty so the system knows when it is guessing vs confident.
- The outputs land in `data/qml/qsvr_metrics.json` and `data/qml/qgpr_metrics.json`, giving a continuously updated “physics intuition” the rest of the loop can query.

## 3. Generate new alloy ideas with a quantum-inspired sampler
- **Property-conditioned QGAN**: `scripts/qgan_prototype.py` and `scripts/qgan_property_conditioning.py` sample fresh alloy recipes while respecting MDIA constraints and desired property windows. After conditioning, 100% of suggestions pass the HEA feasibility checks.
- **Candidate library**: The generator writes structured tables such as `data/qml/qgan_conditioned_candidates.csv`, including predicted densities/targets so downstream components can score each idea.

## 4. Orchestrate a digital active-learning loop
- **Acquisition logic**: `scripts/qal_orchestrator.py` ranks candidates by combining QSVR/QGPR predictions (how promising is the alloy?) with uncertainty (how much could we still learn?).
- **Scheduling**: `scripts/qal_closed_loop.py` simulates the real workflow: generate → score → pick top items → send to DFT → absorb feedback → retrain. This loop already demonstrated ~32% fewer labels to reach the same accuracy as a classical baseline.
- **Oversight**: PDA/ALOA agents monitor label efficiency (`data/architecture/label_efficiency_metrics.json`) to ensure the system keeps improving faster than brute-force screening.

## 5. Send only the best ideas to DFT “truth serum”
- **Handoff packages**: For each selected candidate, `scripts/dft/run_dft_workflow.py` or `scripts/hpc/run_real_dft_campaign.py` build a folder with `structure.cif`, `metadata.json`, and Quantum ESPRESSO settings (see `docs/interfaces/dft_handoff.md`).
- **Real simulations**: QE jobs run via the managed queue, either locally or on HPC (`hpc/scripts/t5r4_real_campaign.sbatch`). Results land under `data/dft_workflow/` with MLflow metrics so BRA and RKMA can validate them.
- **Feedback loop**: Once DFT returns densities/energies, the data flows back into the training pool, sharpening the QSVR/QGPR models and updating acquisition priorities for the next cycle.

## 6. Why it matters
- **Fewer expensive shots**: The quantum-aware models and active-learning policies prune the search space before any DFT cycles spin up, lifting label efficiency from the 0.30 target to 0.80 in the latest campaign.
- **Faster discovery**: Each iteration produces a reviewed “shortlist” (`data/dft_workflow/campaigns/<campaign_id>/candidate_library.csv`) so materials scientists see only the most promising HEAs.
- **Traceable progress**: Every decision—data prep, model training, DFT run—is logged with artefacts so reviewers (or future team members) can reproduce the exact path to any candidate.

In short, the scouting system mixes quantum ML intuition with a disciplined active-learning loop so we only pay the DFT bill for alloys that look like winners, accelerating the hunt for tough, lightweight materials.
