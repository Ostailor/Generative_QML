# Tracking Infrastructure

This directory operationalizes continuous monitoring of acceptance criteria through two complementary layers:

1. **Documentation Workspace** (`../workspace`) — Structured records ensuring all tasks satisfy metadata and approval requirements.
2. **Experiment Logger (MLflow)** — Centralized metrics and artefact tracker referenced by acceptance criteria in `TASKS.md`.

## Components
- `mlflow/` — Configuration, scripts, and environment settings for hosting an MLflow tracking server with artifact storage.
- `acceptance_criteria_registry.csv` — Mapping between TASKS.md acceptance criteria and the MLflow fields or documentation artefacts that validate them.
- `reporting/` — Utilities for generating weekly dashboards surfaced in `workspace/registers/status_snapshot.md`.

## Operational Flow
1. Start MLflow services (see `mlflow/README.md`).
2. Each experiment pipeline logs runs with task-specific tags and metrics defined in `acceptance_criteria_registry.csv`.
3. RKMA executes reporting notebooks/scripts to update status snapshots and provenance graphs.
4. PDA reviews dashboards before milestone gates are approved.

