# Generative Quantum Machine Learning for Cost-Efficient, High-Strength Materials Discovery

## Abstract
We present a modular quantum-active-learning workflow for high-entropy alloy discovery combining quantum kernel regression, property-conditioned generation, queue-driven production DFT, and benchmark governance. In a fresh production rerun (`t5r4-20260211-fasttrack-221-mw4`), we achieved 12/12 converged candidates and a 0.90 label-efficiency gain against a 0.30 target while preserving reproducibility and traceability across data, model, and DFT artefacts.

## 1. Introduction
Computational alloy discovery is constrained by expensive first-principles labels. We address this with a closed-loop workflow that prioritizes which candidates receive DFT evaluation and continuously retrains quantum-aware surrogates using newly acquired labels.

## 2. Methods
### 2.1 Data and Constraints
Curated perovskite/HEA/SAA datasets with release manifests and QA checks were used to train regressors and generators.

### 2.2 Quantum Regressors
QSVR and QGPR models provided prediction and uncertainty signals for active candidate ranking.

### 2.3 Property-Conditioned Generation
A constrained generator produced HEA candidates compliant with chemistry and property windows before DFT dispatch.

### 2.4 Active Learning + Production DFT
The workflow integrates queue-managed Quantum ESPRESSO execution with automated retry/latency tracking and campaign-level manifesting.

## 3. Results (Real-Backed Headline Evidence)
1. Real campaign completion: 12/12 completed jobs, 12 valid candidates.
2. Label-efficiency gain: 0.90 (`>= 0.30` gate).
3. M7 benchmarking: positive quantum-vs-classical gap and recorded sensitivity index.
4. M6 pilots: multi-backend adapter runs captured fidelity/cost envelopes used for benchmark context.

## 4. Robustness and Benchmarking
Robustness sweeps under perturbation show bounded sensitivity. Quantum-vs-classical comparisons remain positive under real DFT-informed scoring.

## 5. Reproducibility
All campaign artefacts are packaged with checksums (`real_dft_campaign_v1`) and linked via final provenance graph. Reproduction checks pass with no missing files or checksum mismatches.

## 6. Discussion and Limitations
Hardware-adapter evidence in M6 supports operational planning and comparative analysis but does not replace physical synthesis validation. Simulated-only historical artefacts are retained only for ablation context.
The fresh M5-real campaign used a screening profile (`2x2x2 k-grid`, reduced cutoffs, `2x2x1` supercell) for tractable throughput; reference-backed fidelity claims remain anchored to the dedicated T5R.3 validation set.

## 7. Conclusion
The end-to-end workflow closes a production-ready loop from candidate generation to real DFT feedback with strong label-efficiency and reproducible packaging, establishing a concrete path to conference and preprint dissemination.
