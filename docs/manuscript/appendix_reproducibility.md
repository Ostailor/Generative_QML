# Reproducibility Appendix (`T9.5`)

## A. Artefact Index
1. Real campaign release: `data/releases/real_dft_campaign_v1/`
2. M6 hardware pilots: `data/hardware/`
3. M7 benchmarks: `data/benchmarks/m7/`
4. Reproducibility checks: `data/reproducibility/reproduction_report.json`

## B. Regeneration Commands
```bash
cd /path/to/Quanutum_MS_Pipeline
source .venv/bin/activate
python scripts/publication/run_paper_data_pipeline.py \
  --profile paper-grade \
  --paper-grade-fastest \
  --campaign-id t5r4-20260211-fasttrack-221-mw4 \
  --tracking-uri "file://$(pwd)/mlruns"
```
This fail-fast entry point regenerates and validates the paper data package (M5-real to M9), writes
`data/reproducibility/paper_pipeline_report.json`, and exits non-zero on any threshold or build failure.
In `paper-grade` mode it runs the GPU benchmark suite with a default budget of 1,100 runs.
The paper-grade runtime gate is enforced at 7,200 seconds (about 2 hours) by default.
The pipeline defaults to `--paper-grade-fastest`; use `--paper-grade-fast` for a slower per-run preset.
Use `--profile fasttrack --skip-paper-grade-suite` for a lightweight verification run.
Use `--allow-validation-review` only when intentionally bypassing the strict T5R.3 `status=pass` gate.

## C. Figure Provenance
- `fig_real_dft_kpi.png` <- campaign summary (`closed_loop_summary.json`)
- `fig_hardware_cost_fidelity.png` <- M6 hardware summary
- `fig_m7_benchmark_summary.png` <- M7 benchmark summary

## D. Headline Claim Guardrail
Only real-backed outputs from production DFT and benchmark pipelines are used in main-text claims.
Simulation-only artefacts are designated ablation context.
