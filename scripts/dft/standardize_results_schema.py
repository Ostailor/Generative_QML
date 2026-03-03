#!/usr/bin/env python3
"""Backfill standardized schema fields for legacy DFT result files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[2]
DFT_WORKFLOW_DIR = BASE_DIR / "data" / "dft_workflow"
DFT_INPUT_DIR = BASE_DIR / "data" / "dft_handoff" / "input"


def compute_settings_hash(input_dir: Path) -> str:
    sha = hashlib.sha256()
    for name in ("metadata.json", "structure.cif", "pseudopotentials.csv", "vasp_settings.json"):
        path = input_dir / name
        if not path.exists():
            continue
        sha.update(path.name.encode("utf-8"))
        sha.update(b"\n")
        sha.update(path.read_bytes())
    return sha.hexdigest()


def infer_evidence_tier(request_id: str, metadata: Dict[str, Any]) -> str:
    rid = request_id.upper()
    if rid.startswith(("REALCAM", "BENCH", "QUEUE")):
        return "production_dft"
    if rid.startswith("SMOKE"):
        return "smoke_test"
    source = str(metadata.get("source_dataset", "")).lower()
    if "qgan" in source or "hea" in source:
        return "pilot_dft"
    return "simulation"


def enrich_result(request_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata") or {}
    input_dir = DFT_INPUT_DIR / request_id
    settings_hash = compute_settings_hash(input_dir) if input_dir.exists() else payload.get("dft_settings_hash", "")

    payload.setdefault("schema_version", "2.0.0")
    payload.setdefault(
        "engine",
        {
            "name": "quantum_espresso",
            "command": "pw.x",
            "mode": metadata.get("calculation", "scf"),
        },
    )
    payload.setdefault("dft_settings_hash", settings_hash)
    payload.setdefault("evidence_tier", infer_evidence_tier(request_id, metadata))
    payload.setdefault(
        "uncertainty",
        {
            "formation_energy_eV": 0.0,
            "exp_density_g_cm3": 0.02,
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize schema fields in DFT results")
    parser.add_argument(
        "--workflow-root",
        type=Path,
        default=DFT_WORKFLOW_DIR,
        help="Root containing <request_id>/results.json directories.",
    )
    args = parser.parse_args()

    updated = 0
    scanned = 0
    for results_path in sorted(args.workflow_root.glob("*/results.json")):
        scanned += 1
        request_id = results_path.parent.name
        payload = json.loads(results_path.read_text())
        before = json.dumps(payload, sort_keys=True)
        payload = enrich_result(request_id, payload)
        after = json.dumps(payload, sort_keys=True)
        if before != after:
            results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            updated += 1

    summary = {"scanned": scanned, "updated": updated, "root": str(args.workflow_root)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
