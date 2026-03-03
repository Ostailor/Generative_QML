# Automated DFT Workflow Plan — T5.1

## Steps
1. Fetch input package from `data/dft_handoff/input/<request_id>/`.
2. Execute Quantum ESPRESSO production workflow via `scripts/dft/run_dft_workflow.py` and produce outputs in `data/dft_workflow/<request_id>/`.
3. Append queue/run summaries to `data/dft_workflow/workflow_report.json` and campaign monitors under `data/dft_workflow/campaigns/<campaign_id>/`.
4. Log metrics via MLflow (latency, completion/failure counts, energies, acceptance hooks).

## Production Notes
- Queue execution supports retries, latency tracking, and MLflow logging (`T5R.2`).
- Campaign-aware validation is handled by `scripts/dft/validate_production_outputs.py --campaign-id <id>`.
- Result payloads are schema-versioned (`schema_version: 2.0.0`) with `engine`, `dft_settings_hash`, `uncertainty`, and `evidence_tier`.
