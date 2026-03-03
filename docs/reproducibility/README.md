# M8 Reproducibility Package

This package consolidates reproducibility artefacts for `M8` closure.

## Core Assets
- `scripts/repro/run_repro_check.py`
- `data/reproducibility/reproduction_report.json`
- `data/reproducibility/provenance_graph_final.json`
- `docs/reproducibility/transparency_and_ethics.md`
- `docs/reproducibility/Dockerfile`
- `docs/reproducibility/requirements-lock.txt`

## Acceptance Mapping
- `T8.1`: provenance graph finalized and countable node set.
- `T8.2`: reproducibility check run with `rkma.reproduction_success == True`.
- `T8.3`: transparency and ethics statement archived.
- `T8.4`: readiness checklist feeds manuscript submission package.

## Reproduction Entry Point
```bash
cd /path/to/Quanutum_MS_Pipeline
source .venv/bin/activate
python scripts/publication/run_paper_data_pipeline.py \
  --profile paper-grade \
  --paper-grade-fastest \
  --campaign-id t5r4-20260211-fasttrack-221-mw4 \
  --tracking-uri "file://$(pwd)/mlruns"
```
Paper-grade mode executes the GPU benchmark suite at a default target of 1,100 runs.
It enforces a 7,200-second (about 2 hours) runtime gate by default.
By default the suite uses `--paper-grade-fastest`; switch to `--paper-grade-fast` for a slower per-run preset.
Append `--allow-validation-review` only when you explicitly want to continue past a T5R.3 `status=review` outcome.
Append `--paper-grade-skip-runtime-gate` only for debugging.
Use `--profile fasttrack --skip-paper-grade-suite` for a shorter non-paper-grade validation.
