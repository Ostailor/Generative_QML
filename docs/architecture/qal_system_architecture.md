# QAL Loop Architecture — M4 T4.1

## Components
1. **Data Layer**: Processed datasets (`hea_features`, `perovskites_features`, `saa_features`) and noise simulations.
2. **Model Layer**:
   - Quantum regressors (QSVR, QGPR) with feature maps from T2.1.
  - Quantum generator (QGAN, T3.2–T3.3) with property conditioning.
   - Classical baselines for benchmarking.
3. **Acquisition Engine**:
   - Uncertainty/Bayesian policies (classical baselines).
   - Quantum-enhanced acquisition (future T4.2).
4. **DFT Integration**:
   - Input/output contract defined in `docs/interfaces/dft_handoff.md`.
   - Asynchronous feedback merges DFT results into labeled pool.
5. **Orchestrator**:
   - Workflow coordinating generation → evaluation → DFT → retraining.
   - Logging via MLflow and provenance updates (RKMA).

## Data Flow (Sequence)
```
Generator → Candidate Filtering → Active Learning Engine →
    ├─ Query Selection → DFT Handoff → DFT Results → Model Update
    └─ Model Update → Metrics Logging → Provenance Update
```

## APIs & Contracts
- **Generator API**: `generate_candidates(n, target_properties)` returns candidate DataFrame.
- **Evaluator API**: `evaluate_candidates(candidates)` returns scoring metrics.
- **Handoff API**: `submit_dft_request(package)` / `fetch_dft_results(request_id)` per T1.5.

## Resource Considerations
- Quantum circuits transpiled for IonQ Harmony (depth ≤40) and IBM Perth (depth ≤60).
- Parallelized DFT submissions to hide latency; queue management to be implemented in T4.3.

## Next Steps
- Implement orchestrator scripts and acceptance hooks (T4.2/T4.3).
- Integrate CI checks ensuring architecture doc stays in sync.

