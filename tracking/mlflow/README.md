# MLflow Tracking Configuration

This setup provides a self-hosted MLflow server that all agents use to log experiments, metrics, and artefacts required by acceptance criteria in `TASKS.md`.

## Directory Structure
- `tracking/mlflow/mlruns/` — Default artifact store (local filesystem). For production, configure S3/MinIO via environment variables.
- `tracking/mlflow/docker-compose.yaml` — Optional containerized deployment using PostgreSQL + MinIO.
- `tracking/mlflow/start_server.sh` — Convenience script to launch a local MLflow server with file-based backend and port availability checks.
- `tracking/mlflow/setup.env.example` — Environment variable template to share credentials and endpoints across agents.

## Quick Start (Local Development)
1. Create and activate a Python environment with `mlflow>=2.9`.
2. Populate `setup.env` from the example template and source it.
3. Ensure the desired port is free (default `5000`). Set `PORT=<free-port>` if another service is listening.
4. Run `./start_server.sh` to start MLflow at `http://localhost:$PORT` with artefacts stored in `tracking/mlflow/mlruns`.
5. Configure clients by exporting `MLFLOW_TRACKING_URI=http://localhost:$PORT`.
6. Use experiment names matching milestones (e.g., `M2_QSVR_Studies`). Ensure each run logs tag `task:<task_id>` and metrics listed in `tracking/acceptance_criteria_registry.csv`.

## Production Deployment (Optional)
- Use `docker-compose up -d` to start a persistent MLflow instance backed by PostgreSQL (metrics) and MinIO (artifacts).
- Update `setup.env` with database credentials and S3 settings.
- Configure reverse proxy / TLS as needed for collaboration.

## Logging Conventions
- **Tags**: `task`, `milestone`, `agent`, `dataset_version`, `code_revision`.
- **Params**: hyperparameters, acquisition strategies, DFT configuration IDs, target thresholds (e.g., `target_accuracy`).
- **Metrics**: Must include variables referenced in `target_expression` definitions.
- **Artifacts**: Upload figures, tables, notebooks, and hashed dataset manifests.

## Health Checks
- Agents verify connectivity by running `mlflow experiments list` and logging a dummy run before starting milestone work.
- RKMA monitors server uptime and storage consumption weekly.

