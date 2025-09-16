#!/usr/bin/env python3
"""Validate DFT handoff packages for T1.5."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

REQUIRED_INPUT_FILES = ["metadata.json", "structure.cif", "vasp_settings.json", "pseudopotentials.csv"]
REQUIRED_OUTPUT_FILES = ["results.json", "structure_relaxed.cif", "log.txt"]


@dataclass
class PackageReport:
    request_id: str
    input_valid: bool
    output_valid: bool
    issues: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "request_id": self.request_id,
            "input_valid": self.input_valid,
            "output_valid": self.output_valid,
            "issues": self.issues,
        }


def compute_settings_hash(input_dir: Path) -> str:
    sha = hashlib.sha256()
    for name in REQUIRED_INPUT_FILES:
        path = input_dir / name
        sha.update((name + "\n").encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def validate_package(input_dir: Path, output_dir: Path) -> PackageReport:
    issues: List[str] = []
    input_valid = True
    output_valid = True

    for name in REQUIRED_INPUT_FILES:
        if not (input_dir / name).exists():
            issues.append(f"Missing input file: {name}")
            input_valid = False

    metadata_path = input_dir / "metadata.json"
    request_id = input_dir.name
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        request_id = meta.get("request_id", request_id)
        required_fields = ["source_dataset", "timestamp_utc", "composition"]
        for field in required_fields:
            if field not in meta:
                issues.append(f"Metadata missing field: {field}")
                input_valid = False
    else:
        input_valid = False

    settings_hash = None
    if input_valid:
        settings_hash = compute_settings_hash(input_dir)

    for name in REQUIRED_OUTPUT_FILES:
        if not (output_dir / name).exists():
            issues.append(f"Missing output file: {name}")
            output_valid = False

    results_path = output_dir / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        if settings_hash and results.get("dft_settings_hash") != settings_hash:
            issues.append("dft_settings_hash mismatch")
            output_valid = False
    else:
        output_valid = False

    return PackageReport(request_id=request_id, input_valid=input_valid, output_valid=output_valid, issues=issues)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DFT handoff packages")
    parser.add_argument("--input-root", default="data/dft_handoff/input")
    parser.add_argument("--output-root", default="data/dft_handoff/output")
    parser.add_argument("--report", default="data/metadata/qa_reports/dft_handoff_validation.json")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    reports: List[PackageReport] = []
    for request_dir in sorted(input_root.iterdir()):
        if not request_dir.is_dir():
            continue
        output_dir = output_root / request_dir.name
        report = validate_package(request_dir, output_dir)
        reports.append(report)

    summary = {
        "total_packages": len(reports),
        "input_valid": sum(r.input_valid for r in reports),
        "output_valid": sum(r.output_valid for r in reports),
        "reports": [r.to_dict() for r in reports],
    }

    Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
