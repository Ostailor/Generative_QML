# 2026-02-11 — M5-real Full Re-Run (Fresh Execution)

## Context
User requested a fresh, end-to-end execution of `M5-real` with real solver runs in this session, plus publication-ready evidence.

## Fresh runs executed
1. `T5R.1` QE benchmark (real run)
- Command: `scripts/dft/qe_benchmark.py --request-id QAL-0001 ...`
- MLflow run: `b95a3af3e1834d1ea9b4141ddf21ba62` (latest task run)
- Benchmark wall time: `190.96 s`

2. `T5R.2` queue integration (real run)
- Command: `scripts/dft/run_dft_workflow.py --queue QUEUE-0001 QUEUE-0002 QUEUE-0003 ...`
- Queue run id: `dft-queue-20260211T025751`
- Result: `3/3 completed`, `0 failed`, latency avg `223.25 s`
- MLflow task run: `71c3f644e5fa4943ab6395558871e36b`

3. `T5R.3` validation versus references (real run)
- Command: `scripts/dft/validate_production_outputs.py --requests QAL-0001 QUEUE-0001 QUEUE-0002 QUEUE-0003 BENCH-RELAX-0001 ...`
- MLflow run: `83fafa5d6f5e4ee2897ce0a3614ddb84`
- `bra.dft_validation_gap = 0.03393` (passes `<= 0.05`)

4. `T5R.4` real campaign (fresh run)
- Campaign id: `t5r4-20260211-fasttrack-221-mw4`
- Command: `scripts/hpc/run_real_dft_campaign.py --iterations 3 --top-k 4 --max-workers 4 ...`
- Runtime profile used for tractability: `k-grid 2x2x2`, `ecutwfc 28 Ry`, `ecutrho 224 Ry`, `supercell 2x2x1`
- MLflow run: `c05c96f0e7f641cc9170ea8598711acc`
- Result: `12/12 completed`, `0 failed`, `valid_candidates = 12`, `label_efficiency_gain = 0.90`

5. `T5R.5` release package (fresh run)
- Command: `scripts/releases/create_real_dft_release.py --campaign-id t5r4-20260211-fasttrack-221-mw4 --release-dir data/releases/real_dft_campaign_v3_20260211`
- Release path: `data/releases/real_dft_campaign_v3_20260211`
- MLflow run: `20d2a96f62d544109dd642886849c5e7`

## Code-level fixes applied during rerun
1. Isolated QE run directories in queue workers to avoid `espresso.pwi/pwo` collision under parallel execution:
- `scripts/dft/run_dft_workflow.py`

2. Added campaign supercell configurability for runtime-quality tuning:
- `scripts/hpc/run_real_dft_campaign.py` (`--supercell`)

3. Hardened validation semantics to avoid false pass when no references are matched:
- `scripts/dft/validate_production_outputs.py`
- Added `referenced_formulas` + `missing_reference_count` and stricter `status` logic.

## Acceptance state after rerun
From `workspace/registers/status_snapshot.md` (regenerated on 2026-02-11):
- `T5R.1`: pass
- `T5R.2`: pass
- `T5R.3`: pass
- `T5R.4`: pass
- `T5R.5`: pass

## Notes
- A campaign-level validation report for the new fasttrack compositions was generated and correctly marked `review` because the reference parquet does not contain those exact formulas.
- Headline validation claim remains anchored to the reference-backed `T5R.3` run above.
