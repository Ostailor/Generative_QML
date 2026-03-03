# M7 Metric Protocol (`T7.1`)

## Objective
Define a reproducible, statistically explicit metric suite for quantum-vs-classical comparisons under real DFT and hardware-aware conditions.

## Primary Metrics
1. `bra.quantum_vs_classical_gap`
2. `bra.sensitivity_index`
3. `bra.dft_delta_vs_classical_rmse`
4. `aloa.real_label_efficiency_gain`
5. `bra.hardware_mean_fidelity`

## Statistical Policy
1. Use fixed seeds for perturbation sweeps.
2. Report central tendency + variability for each benchmark metric.
3. Treat simulated-only metrics as ablation context; do not use for headline claims.

## Acceptance Hook
- `bra.metric_protocol_ready = 1`
