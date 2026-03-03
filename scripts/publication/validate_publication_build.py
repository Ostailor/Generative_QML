#!/usr/bin/env python3
"""Validate publication build pipeline: figures, bibliography, manuscript bundle, poster export."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = BASE_DIR / "docs" / "manuscript"
POSTER_DIR = BASE_DIR / "docs" / "poster"
SUBMISSION_BUILD_DIR = BASE_DIR / "docs" / "submission" / "build"
REFERENCES_BIB = MANUSCRIPT_DIR / "references.bib"
BUILD_REPORT = SUBMISSION_BUILD_DIR / "publication_build_report.json"
DEFAULT_CAMPAIGN_ID = "t5r4-20260211-fasttrack-221-mw4"
DEFAULT_CAMPAIGN_ROOT = BASE_DIR / "data" / "dft_workflow" / "campaigns"

EXPECTED_FIGURES = {
    "fig_real_dft_kpi.png",
    "fig_hardware_cost_fidelity.png",
    "fig_m7_benchmark_summary.png",
}


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)


def _parse_bib_keys(path: Path) -> List[str]:
    if not path.exists():
        return []
    keys: List[str] = []
    pattern = re.compile(r"@\w+\{([^,]+),")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line.strip())
        if match:
            keys.append(match.group(1).strip())
    return keys


def _compile_manuscript_bundle() -> Dict[str, str]:
    SUBMISSION_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    paper = MANUSCRIPT_DIR / "conference_paper.md"
    appendix = MANUSCRIPT_DIR / "appendix_reproducibility.md"
    output = SUBMISSION_BUILD_DIR / "conference_paper_submission.md"

    body = []
    body.append("# Submission Bundle")
    body.append("")
    body.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    body.append("")
    body.append(paper.read_text(encoding="utf-8"))
    body.append("")
    body.append("## Reproducibility Appendix")
    body.append("")
    body.append(appendix.read_text(encoding="utf-8"))
    output.write_text("\n".join(body), encoding="utf-8")
    return {"bundle": str(output)}


def _build_bibliography() -> Dict[str, object]:
    keys = _parse_bib_keys(REFERENCES_BIB)
    out = SUBMISSION_BUILD_DIR / "bibliography_index.json"
    payload = {
        "source": str(REFERENCES_BIB),
        "entries": sorted(keys),
        "entry_count": len(keys),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"index": str(out), "entry_count": len(keys)}


def _ensure_figures(campaign_id: str, campaign_root: Path) -> Dict[str, object]:
    _run(
        [
            sys.executable,
            "scripts/publication/build_figures.py",
            "--campaign-id",
            campaign_id,
            "--campaign-root",
            str(campaign_root),
        ]
    )
    fig_dir = MANUSCRIPT_DIR / "figures"
    present = {p.name for p in fig_dir.glob("*.png")}
    missing = sorted(EXPECTED_FIGURES - present)
    return {"figure_dir": str(fig_dir), "missing_figures": missing, "count": len(present)}


def _export_poster() -> Dict[str, str]:
    _run([sys.executable, "scripts/publication/create_poster_manifest.py"])
    manifest = POSTER_DIR / "poster_manifest.json"
    export_dir = SUBMISSION_BUILD_DIR / "poster_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(POSTER_DIR / "conference_poster.md", export_dir / "conference_poster.md")
    shutil.copy2(manifest, export_dir / "poster_manifest.json")
    return {"poster_manifest": str(manifest), "poster_export_dir": str(export_dir)}


def run_build(campaign_id: str = DEFAULT_CAMPAIGN_ID, campaign_root: Path = DEFAULT_CAMPAIGN_ROOT) -> Dict[str, object]:
    compile_result = _compile_manuscript_bundle()
    bibliography_result = _build_bibliography()
    figures_result = _ensure_figures(campaign_id=campaign_id, campaign_root=campaign_root)
    poster_result = _export_poster()

    success = bibliography_result["entry_count"] > 0 and len(figures_result["missing_figures"]) == 0
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass" if success else "fail",
        "campaign_id": campaign_id,
        "steps": {
            "manuscript_compile": compile_result,
            "bibliography_build": bibliography_result,
            "figure_regeneration": figures_result,
            "poster_export": poster_result,
        },
    }
    BUILD_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate publication build pipeline.")
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    args = parser.parse_args()
    report = run_build(campaign_id=args.campaign_id, campaign_root=args.campaign_root)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
