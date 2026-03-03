# M7 Benchmark and Robustness Report

## Inputs
- `/Users/omtailor/Quanutum_MS_Pipeline/data/dft_workflow/campaigns/t5r4-14539888/closed_loop_summary.json`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/dft_workflow/campaigns/t5r4-14539888/candidate_library.csv`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/qml/classical_al_metrics.json`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/qml/qsvr_metrics.json`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/qml/qgpr_metrics.json`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/hardware/hardware_summary.json`

## Results
- `bra.sensitivity_index`: `0.3090`
- `bra.quantum_vs_classical_gap`: `0.5000`
- `bra.dft_delta_vs_classical_rmse`: `0.1931`
- `aloa.real_label_efficiency_gain`: `0.8000`

## Interpretation
1. Robustness perturbation response remained bounded and quantifiable (`sensitivity_index` recorded).
2. Quantum workflow preserved a positive benchmark gap against classical baseline under real DFT-informed evaluation.
3. Hardware fidelity remained high enough to support cross-backend benchmark interpretation.

## Artefacts
- `/Users/omtailor/Quanutum_MS_Pipeline/data/benchmarks/m7/m7_summary.json`
- `/Users/omtailor/Quanutum_MS_Pipeline/data/benchmarks/m7/m7_metrics_table.csv`

## Claim Boundary
Primary conclusions rely on production DFT campaign evidence and hardware-adapter metrics.
Legacy mock/simulated-only runs are not used for headline claims.
