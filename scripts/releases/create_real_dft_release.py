#!/usr/bin/env python3
"""Create release bundle for the production real DFT campaign (T5R.5)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = BASE_DIR / "data" / "dft_workflow" / "campaigns"
RELEASE_ROOT = BASE_DIR / "data" / "releases"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def build_manifest(root: Path) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "release_manifest.json":
            # Avoid self-referential checksum drift when rewriting the manifest.
            continue
        rel = path.relative_to(root)
        manifest.append(
            {
                "path": str(rel),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return manifest


def write_release_docs(out_dir: Path, campaign_id: str, summary: Dict[str, object]) -> None:
    readme = f"""# Real DFT Campaign Release v1

This package closes `T5R.5` by freezing the production campaign artefacts for `{campaign_id}`.

## Contents
- `campaign/` — Full campaign snapshot copied from `data/dft_workflow/campaigns/{campaign_id}`.
- `release_info.json` — High-level release metadata and headline real-only metrics.
- `release_manifest.json` — SHA256 + size inventory for reproducibility.
- `reproduction_runbook.md` — Deterministic replay and validation instructions.

## Evidence Policy
This release is **real-backed**: headline claims are derived from production DFT and queue outputs.
Simulated/mock artefacts are excluded from primary claims.
"""
    runbook = f"""# Reproduction Runbook (Real DFT Campaign v1)

## 1) Environment
1. Activate the project environment.
2. Export `MLFLOW_TRACKING_URI=file://{(BASE_DIR / 'mlruns').resolve()}`.

## 2) Validate campaign artefacts
Run:
```bash
python scripts/dft/validate_production_outputs.py --campaign-id {campaign_id}
```

## 3) Verify release checksums
Run:
```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('data/releases/real_dft_campaign_v1')
manifest = json.loads((root/'release_manifest.json').read_text())
for entry in manifest:
    path = root / entry['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry['sha256'], f"checksum mismatch: {{entry['path']}}"
print('checksum verification passed')
PY
```

## 4) Refresh acceptance tracking
Run:
```bash
python tracking/reporting/update_status_snapshot.py --mlflow-tracking-uri file://$(pwd)/mlruns
```

## 5) Expected headline values
- `label_efficiency_gain`: {summary.get('label_efficiency_gain')}
- `valid_candidates`: {summary.get('valid_candidates')}
- `completed_jobs`: {summary.get('completed_jobs')}
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "reproduction_runbook.md").write_text(runbook, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create real DFT release bundle")
    parser.add_argument("--campaign-id", default="t5r4-14539888")
    parser.add_argument("--release-dir", type=Path, default=RELEASE_ROOT / "real_dft_campaign_v1")
    args = parser.parse_args()

    campaign_dir = CAMPAIGN_ROOT / args.campaign_id
    if not campaign_dir.exists():
        raise SystemExit(f"Campaign not found: {campaign_dir}")

    out_dir = args.release_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_campaign_dir = out_dir / "campaign"
    copy_tree(campaign_dir, target_campaign_dir)

    summary_path = target_campaign_dir / "closed_loop_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    release_info = {
        "version": "1.0.0",
        "release_date_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "task": "T5R.5",
        "milestone": "M5-real",
        "campaign_id": args.campaign_id,
        "evidence_tier": "production_dft",
        "headline_metrics": {
            "label_efficiency_gain": summary.get("label_efficiency_gain"),
            "valid_candidates": summary.get("valid_candidates"),
            "completed_jobs": summary.get("completed_jobs"),
            "failed_jobs": summary.get("failed_jobs"),
        },
        "sources": [
            str(campaign_dir),
            "scripts/dft/validate_production_outputs.py",
            "scripts/dft/run_dft_workflow.py",
            "scripts/hpc/run_real_dft_campaign.py",
        ],
    }
    (out_dir / "release_info.json").write_text(json.dumps(release_info, indent=2), encoding="utf-8")

    write_release_docs(out_dir, args.campaign_id, release_info["headline_metrics"])

    manifest = build_manifest(out_dir)
    (out_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "release_dir": str(out_dir),
                "files": len(manifest),
                "campaign_id": args.campaign_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
