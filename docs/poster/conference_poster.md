# Conference Poster Draft

## Title
Generative Quantum Machine Learning for Cost-Efficient, High-Strength Materials Discovery

## Panel 1: Problem and Approach
- DFT labels are expensive and throughput-limited.
- We combine quantum regressors, constrained generation, and active-learning scheduling.

## Panel 2: Production DFT Campaign
- Campaign: `t5r4-20260211-fasttrack-221-mw4`
- Completed jobs: `12`
- Valid candidates: `12`
- Label-efficiency gain: `0.90` (target `0.30`)

Figure: `fig_real_dft_kpi.png`

## Panel 3: Hardware + Benchmark Context
- M6: 3 backends, 27 runs, mean fidelity ~0.958.
- M7: positive quantum-vs-classical gap with recorded perturbation sensitivity index.

Figures:
- `fig_hardware_cost_fidelity.png`
- `fig_m7_benchmark_summary.png`

## Panel 4: Reproducibility and Next Submission Window
- Release package includes campaign snapshot + checksums + runbook.
- Reproduction check passes (`rkma.reproduction_success = True`).
- Submission path: conference paper + poster + immediate preprint package.
