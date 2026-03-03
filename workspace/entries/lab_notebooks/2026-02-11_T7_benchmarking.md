---
entry_id: BRA-LAB-20260211-001
entry_type: lab_notebook
task_id: T7.4
milestone_id: M7
owning_agent: BRA
collaborators: [ALOA, QHSOA]
dataset_version: BENCH-M7-1.0.0
code_revision: finish-to-submission-2026-02-11
mlflow_run_id: a53cfad6e1714e53af3d5744bc8ece6c,c808f81e3f0341b0b60b204b8c4eb6fe
qpu_backend: multi-backend-adapter
submission_status: approved
reviewer_comments: T7.2 and T7.4 metrics logged and report archived.
timestamp_utc: 2026-02-11T02:29:00Z
---

## Objective
Run robustness and quantum-vs-classical benchmarking analyses using production campaign outputs.

## Results
- `bra.sensitivity_index`: 0.3090
- `bra.quantum_vs_classical_gap`: 0.5000
- `bra.dft_delta_vs_classical_rmse`: 0.1931

## Artefacts
- `/Users/omtailor/Quanutum_MS_Pipeline/data/benchmarks/m7/m7_summary.json`
- `/Users/omtailor/Quanutum_MS_Pipeline/docs/benchmarking/m7_benchmark_report.md`
- `/Users/omtailor/Quanutum_MS_Pipeline/docs/benchmarking/m7_metric_protocol.md`

## Acceptance Check
Both `T7.2` and `T7.4` acceptance metrics are logged with task-tagged MLflow runs.
