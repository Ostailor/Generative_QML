# QAL Monitoring & Metrics Hooks

## Objectives
- Capture per-iteration metrics (candidates generated, DFT completions, acquisition gains) once DFT integration is active.
- Provide append-only JSONL log for observability dashboards.

## Implementation
- `scripts/qal_orchestrator.py` appends run summaries to `data/architecture/qal_monitoring_log.jsonl` including timestamp, selected candidates, and DFT completion count.
- MLflow runs (T4.3) continue to track aggregate metrics; monitoring log enables streaming ingestion for dashboards (e.g., Prometheus/ELK).

## Next Steps
- Integrate with production logging pipeline post-M5.
- Extend log schema with latency measurements once async DFT queue is implemented.

