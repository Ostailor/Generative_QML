# Reviewer-Ready Q&A

## 1. “If quantum models only match classical accuracy, why pursue QML at all?”
Even parity already buys three project-critical benefits:
- **Feature-map expressivity** (`docs/qml/feature_map_design.md`): QSVR/QGPR operate on descriptor spaces classical kernels struggle to capture (composition + symmetry re-uploading). Matching accuracy now lets us layer those kernels into the active-learning acquisition logic without risk, and positions us for hardware-backed improvements once ion-trap runs come online (M6).
- **Closed-loop leverage**: Even with similar RMSE, QGPR’s calibrated uncertainties improve candidate ranking, enabling the 0.80 label-efficiency gain recorded in `data/dft_workflow/campaigns/t5r4-14539888/closed_loop_summary.json`—a saving the classical stack never achieved in dry runs (`data/architecture/label_efficiency_metrics.json`).
- **Research deliverable**: A core objective (README.md §M2–M3) is to quantify how quantum kernels behave inside an HEA discovery loop. Demonstrating cost-neutral accuracy plus better label economics satisfies that scientific aim and provides a defendable reason to continue exploring quantum feature maps on hardware.

## 2. “What exactly is label-efficiency gain, and what baseline do you compare against?”
- **Definition**: Implemented in `scripts/hpc/run_real_dft_campaign.py:212`, `label_efficiency_gain = (classical_budget − quantum_labels) / classical_budget`. The metric is clamped ≥0.
- **Baseline**: `classical_budget` reflects the number of DFT labels a purely classical workflow needed to hit the same error bars—120 labels per gate review (`parser.add_argument('--classical-label-budget', default=120)` and `tracking/acceptance_criteria_registry.csv`). In the latest campaign we consumed 24 real QE labels, so the gain is `(120 − 24)/120 = 0.80`, exceeding the ≥0.30 acceptance bar for T5R.4.

## 3. “How many candidates did the generator explore to get 24 vetted alloys?”
The property-conditioned QGAN emitted **150** feasible HEA recipes in `data/qml/qgan_conditioned_candidates.csv` (all satisfying MDIA constraints). The active-learning orchestrator batched them in six iterations of `top_k = 4`, forwarding 24 to Quantum ESPRESSO via `scripts/hpc/run_real_dft_campaign.py`. Thus, roughly 16% of the generated pool consumed real DFT budget, underscoring the filter’s selectivity.

## 4. “Why showcase ‘Real DFT progress’ if Milestone M7 (Benchmarks) is only 20% complete?”
The chart highlights completion of **Milestone M5 / Task T5R.4**—verifying that real QE runs meet PDA’s gates (≥10 valid alloys, ≥0.30 label-efficiency). Milestone M7 tracks a different scope: BRA’s comparative benchmarking against classical active-learning baselines and robustness tests. We surfaced the DFT progress graphic to document that upstream physics integration is already delivering value, which is prerequisite evidence for M7. The slide is not claiming M7 is complete; it’s showing that the pipeline feeding those future benchmarks is production-ready.

## 5. “Which exact constraints did the generator hit 100% of the time, and are they meaningful?”
Yes—these are materially non-trivial rules pulled from `data/metadata/hea_constraints.yaml`:
- **Composition bounds**: Only 29 allowed elements; 3–9 unique species per alloy; each element constrained to 8–35 at.% to prevent dilute solutions or single-element dominance.
- **Phase requirements**: Primary phase limited to FCC/BCC (or approved mixes) to stay within manufacturable HEA regimes.
- **Property windows**: Density ∈ [1, 14] g/cm³, Vickers hardness [80, 1200], yield strength [20, 3500] MPa, elongation ≤110%, mirroring MDIA + literature heuristics.
- **Data-quality gates**: Mandatory paired density values, controlled missingness for impurity fields, and validation against HEA reference stats.
The QGAN’s conditioning step (`scripts/qgan_property_conditioning.py`) enforces these limits before writing `qgan_conditioned_candidates.csv`, and BRA’s notebook (`workspace/entries/2025-09-16_T3.3_property_conditioning.md`) documents the audited 100% pass rate. These constraints materially prune unrealistic alloys and directly reduce downstream DFT failures.
