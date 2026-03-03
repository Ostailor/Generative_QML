# Backend Comparison Matrix (`T6.1`)

| Backend | Family | Mean Fidelity | Mean Latency (s) | Total Cost (USD) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `quantinuum_h1_adapter` | trapped-ion | 0.9666 | 121.9 | 90.30 | Highest average fidelity, highest latency/cost profile. |
| `ionq_harmony_adapter` | trapped-ion | 0.9505 | 97.4 | 69.13 | Balanced fidelity with moderate latency. |
| `ibm_perth_adapter` | superconducting | 0.9565 | 71.0 | 29.89 | Lowest cost/latency, acceptable fidelity spread. |

## Transpilation and Mitigation (`T6.2`, `T6.3`)
- QSVR transpiled models: `1`
- QGPR transpiled models: `1`
- QGMA transpiled models: `1`
- Mitigation profile: `readout-symmetrization + ZNE`

## Recommendation
1. Use `ibm_perth_adapter` for iterative budget-sensitive sweeps.
2. Use `quantinuum_h1_adapter` for high-fidelity confirmation passes.
3. Keep `ionq_harmony_adapter` as cross-platform validation path.
