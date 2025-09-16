# QSVR/QGPR Experiment Documentation — M2 T2.5

## Run Overview
| Task | MLflow Run ID | Artefacts |
| --- | --- | --- |
| T2.2 QSVR Benchmark | aee97d58ddc0416fb13bdc6628571635 | `data/qml/qsvr_metrics.json`, `data/qml/qsvr_predictions.csv` |
| T2.3 QGPR Benchmark | b0c579003f7b4fd3a163be299fa82f8b | `data/qml/qgpr_metrics.json` |
| T2.4 Classical Baseline | b15d4a0bf60842b583f7d1d4a5ebb0c9 | `data/qml/classical_al_metrics.json` |

## Key Metrics
- QSVR relative RMSE gap: −1.64e-05 (within 10% tolerance).
- QGPR coverage: 0.936 (gap −0.014 within ±0.05 requirement).
- Classical AL final RMSE: 1.496.

## Provenance
See `data/qml/provenance_graph.json` for mapping between runs, artefacts, and source datasets (Dataset v1.0).

## Sign-off
- RKMA reviewed metadata compliance and recorded provenance update (2025-09-16).
- PDA acknowledged documentation completeness for T2.5 milestone.

