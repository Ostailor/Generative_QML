# Automated DFT Workflow Plan — T5.1

## Steps
1. Fetch input package from `data/dft_handoff/input/<request_id>/`.
2. Execute simulation (mocked for now) and produce outputs in `data/dft_workflow/<request_id>/`.
3. Append run summary to `data/dft_workflow/workflow_report.json`.
4. Metrics logged via MLflow (run count, energy stats).

## Future Enhancements
- Integrate real DFT engine (Quantum ESPRESSO/ VASP) using workflow manager (e.g., FireWorks).
- Add error handling, retry logic, and latency metrics.
- Connect to monitoring pipeline for observability.

