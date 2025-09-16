# Reporting Toolkit

RKMA uses these assets to generate continuous monitoring outputs.

## Pending Assets
- `status_snapshot.ipynb` (to be created) — Pulls MLflow statistics and updates `workspace/registers/status_snapshot.md`.
- `risk_register_updater.py` (optional) — Syncs risks captured in decision logs.

## Data Sources
- MLflow REST API (default `http://localhost:5000`).
- Workspace registers (CSV / Markdown) for onboarding and approvals.

## Operational Cadence
1. Run `python reporting/update_status_snapshot.py --week <ISO week>`.
2. Attach resulting summary to PDA weekly briefing.
3. Archive generated reports under `tracking/reporting/history/` with timestamped filenames.

