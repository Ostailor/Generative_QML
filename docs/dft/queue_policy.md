# DFT Queue Policy (T5R.2)

This document formalizes the production queue policy for running Quantum ESPRESSO jobs inside the active learning loop during milestone **M5-real**. The policy is implemented in `scripts/dft/run_dft_workflow.py` (see `run_queue`) and is designed to run on local workstations or HPC front-ends without additional infrastructure.

## Scheduling Model
- **Submission interface**: `python scripts/dft/run_dft_workflow.py --queue <REQUEST ...>` accepts a list of request identifiers matching folders under `data/dft_handoff/input/`.
- **Concurrency**: `--max-workers` controls the size of the worker pool (default `1`). Each worker launches a pw.x process via ASE; safe values on shared login nodes are `1-2`.
- **Isolation**: Each job operates on `data/dft_workflow/<request_id>` and removes any previous outputs before execution to avoid stale artefacts.

## Failure Recovery
- **Retry budget**: `--max-retries` retries failed jobs in-place (default `1`). Backoff uses `min(10.0, 1.5 * attempt)` seconds to avoid hammering shared filesystems.
- **Error capture**: All exceptions (ASE, QE, filesystem) are recorded with tracebacks. Failed jobs terminate after the retry budget is exceeded; no partial results are promoted.
- **Idempotency**: Re-running the queue with the same `request_id` set is safe because each attempt clears its output directory before relaunching.

## Monitoring & Instrumentation
- **Job log**: `data/dft_workflow/queue_runs/<run_id>/jobs.jsonl` contains per-job records (status, attempts, latency, error traces).
- **Summary artefact**: `summary.json` stores aggregate statistics (completed, failed, latency min/avg/max/p95). This file is ingested by the AL orchestrator dashboard.
- **MLflow metrics**: The queue logs `aloa.real_dft_iterations`, `mdia.dft_jobs_completed`, `mdia.dft_jobs_failed`, and latency metrics (`dft_queue.latency_*_s`) under experiment `dft_queue_runs` with tags `{task: T5R.2, milestone: M5-real}`.

## Integration Hooks
- **Active learning loop**: The orchestrator polls `summary.json` and the `jobs.jsonl` stream to update `aloa.real_dft_iterations` and capture latency distributions for scheduling decisions.
- **Alerting**: Non-zero failures trigger BRA review before results flow downstream. The monitoring path is also surfaced in `workspace/entries/lab_notebooks` for reproducibility logging.

## Operational Guidelines
- Validate each new QE binary or pseudopotential set via `scripts/dft/qe_benchmark.py` before promoting to queue runs.
- Keep `--max-workers` aligned with available pw.x licenses/cores; use HPC scheduler wrappers after M5-real if bigger batch throughput is required.
- Archive queue artefacts through RKMA at the end of each production campaign (feeds T5R.5 reproducibility package).
