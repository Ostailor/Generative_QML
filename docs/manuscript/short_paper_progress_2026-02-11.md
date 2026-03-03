# Generative Quantum Active Learning for Cost-Efficient Materials Discovery: Program Status and Fresh Real-DFT Closure (Short Paper)

## Abstract
This report summarizes the current end-to-end status of the "Generative Quantum Machine Learning for Cost-Efficient, High-Strength Materials Discovery" program, with emphasis on fresh real-DFT execution evidence produced on February 11, 2026. A full M5-real rerun was executed in-session using Quantum ESPRESSO production workflows and queue orchestration, yielding complete closure of tasks T5R.1–T5R.5 with fresh MLflow records and release artifacts. The new campaign (`t5r4-20260211-fasttrack-221-mw4`) completed 12/12 DFT jobs with 0 failures, 12 valid candidates, and label-efficiency gain 0.90 against a 0.30 acceptance threshold. Reference-backed physical validation remains positive (`bra.dft_validation_gap = 0.03393`). Beyond M5-real, existing M6–M9 artifacts remain in passing state (hardware benchmarking, robustness reporting, reproducibility packaging, manuscript/poster/preprint bundles), and publication-build checks pass. This establishes a publication-credible, reproducible baseline with explicit claim boundaries: real-backed headline metrics only; simulation-only results constrained to ablation/support context.

## 1. Introduction
The project objective is a modular quantum-active-learning (QAL) loop combining quantum regressors (QSVR/QGPR), property-conditioned generation, and DFT feedback to reduce labeling cost in high-entropy alloy (HEA) discovery while preserving scientific rigor, reproducibility, and benchmark comparability. The immediate operational goal in this phase was to close all remaining publication blockers with fresh evidence, especially uncertainty around whether M5-real reflected historical logs only or current executable reality.

A full fresh rerun was therefore performed for M5-real, and manuscript packaging was upgraded to include an arXiv-ready LaTeX source tree aligned with the real-only claims policy.

## 2. Methods and Execution Design

### 2.1 Program-Control Principle
Execution followed task gates defined in `tracking/acceptance_criteria_registry.csv`, with all closure decisions bound to MLflow-tagged runs and file-level artifacts (campaign summaries, validation reports, release manifests, notebooks, and status snapshots).

### 2.2 Fresh M5-real Execution Sequence
The rerun enforced strict sequencing:
1. T5R.1 (QE readiness benchmark and logging).
2. T5R.2 (real queue orchestration with latency/failure tracking).
3. T5R.3 (reference-backed validation and uncertainty summary).
4. T5R.4 (fresh campaign execution with real DFT jobs).
5. T5R.5 (release packaging and checksum-manifested reproducibility bundle).

### 2.3 Runtime/Throughput Engineering Adjustments
During rerun, several execution bottlenecks were identified and corrected:
- Queue-parallel `pw.x` collision risk was fixed by isolating per-job calculator directories in `scripts/dft/run_dft_workflow.py`.
- Campaign configurability was expanded via `--supercell` in `scripts/hpc/run_real_dft_campaign.py` to make controlled runtime-quality tradeoffs explicit and reproducible.
- Validation semantics were hardened in `scripts/dft/validate_production_outputs.py` so missing reference matches cannot generate false passes (`referenced_formulas`, `missing_reference_count`, stricter status logic).

These are not cosmetic changes; they materially improve correctness and auditability.

## 3. Fresh Results (M5-real)

### 3.1 Task-Level Fresh Run Evidence
- **T5R.1**: `b95a3af3e1834d1ea9b4141ddf21ba62` (`mdia.qe_benchmark_ready = 1.0`, benchmark wall time ~190.96 s).
- **T5R.2**: `71c3f644e5fa4943ab6395558871e36b` (fresh real queue completion with autonomous execution and latency traces).
- **T5R.3**: `83fafa5d6f5e4ee2897ce0a3614ddb84` (`bra.dft_validation_gap = 0.03393`, pass vs. <=0.05 gate).
- **T5R.4**: `c05c96f0e7f641cc9170ea8598711acc` (fresh production campaign closure).
- **T5R.5**: `20d2a96f62d544109dd642886849c5e7` (fresh release documentation signal).

### 3.2 Campaign Outcome (Fresh)
From `data/dft_workflow/campaigns/t5r4-20260211-fasttrack-221-mw4/closed_loop_summary.json`:
- Iterations planned/executed: 3/3.
- Quantum label budget: 12.
- Completed jobs: 12.
- Failed jobs: 0.
- Valid candidates: 12.
- Label-efficiency gain: 0.90.

All values exceed M5-real acceptance targets.

### 3.3 Validation Interpretation
Two validation contexts were preserved intentionally:
- **Reference-backed validation set** (`QAL-0001`, `QUEUE-*`, `BENCH-RELAX-0001`) passed with 3.39% density gap.
- **New fasttrack campaign compositions** are reported with `review` status when references are absent, preventing overclaiming.

This separation is a methodological strength and aligns with the real-only claims boundary.

## 4. M6–M9 Consolidated Status

### 4.1 Hardware/Cost (M6)
`data/hardware/hardware_summary.json` reports:
- 27 hardware runs.
- 3 backends and 3 model types.
- Mean fidelity ~0.9579.
- Total tracked cost ~189.33 USD.
- Mitigation profile includes readout symmetrization + ZNE.

### 4.2 Benchmarking/Robustness (M7)
`data/benchmarks/m7/m7_summary.json` reports:
- `sensitivity_index = 0.3090`.
- `quantum_vs_classical_gap = 0.5000`.
- Positive novelty gap and DFT-vs-classical RMSE delta.

### 4.3 Reproducibility (M8)
`data/reproducibility/reproduction_report.json` reports:
- Required files present: true.
- Release manifest valid: true.
- Reproduction success: true.

### 4.4 Manuscript/Poster/Submission (M9)
- Publication build validation passes (`docs/submission/build/publication_build_report.json`).
- Manuscript bundle, figure regeneration, bibliography index, and poster export all pass automated checks.
- arXiv LaTeX package is present and buildable (`docs/manuscript/arxiv/main.tex`, `main.pdf`).

## 5. Acceptance and Governance State
`workspace/registers/status_snapshot.md` currently reports no failed/missing acceptance entries, including all M5-real rows passing with fresh run IDs. This indicates acceptance-layer closure under the current registry semantics.

## 6. Reproducibility and Artifact Lineage
A fresh release package was generated:
- `data/releases/real_dft_campaign_v3_20260211`

Contents include campaign snapshot, release metadata, checksum manifest, and runbook. This package is suitable for internal audit and external supplementary dissemination.

## 7. Limitations and Claim Scope
1. Fresh M5-real campaign throughput used screening-oriented settings (`k-grid 2x2x2`, reduced cutoffs, `supercell 2x2x1`) to complete full closure under practical runtime constraints.
2. Therefore, primary claims should emphasize pipeline closure, label-efficiency, and reproducibility/completeness; high-precision materials-property claims should continue to anchor to explicitly reference-backed validations.
3. GPU-backed DFT execution was not available in this environment (`nvidia-smi` absent); all fresh DFT runs are real CPU QE runs.

## 8. Conclusion
The project now has a fresh, fully executed M5-real closure with verifiable real-DFT evidence, corrected execution semantics, and updated release/manuscript infrastructure. Combined with existing M6–M9 passing artifacts, this supports immediate conference-prep and preprint dissemination under a defensible real-only headline policy.

## Appendix A: Core Artifact Pointers
- Fresh M5-real campaign summary: `data/dft_workflow/campaigns/t5r4-20260211-fasttrack-221-mw4/closed_loop_summary.json`
- Reference-backed validation report: `data/dft_workflow/validation/validation_report_20260211_refset.json`
- Fresh M5-real release bundle: `data/releases/real_dft_campaign_v3_20260211`
- Fresh M5-real notebook entry: `workspace/entries/lab_notebooks/2026-02-11_M5_real_full_rerun.md`
- Global status snapshot: `workspace/registers/status_snapshot.md`
- arXiv package: `docs/manuscript/arxiv/main.tex`
