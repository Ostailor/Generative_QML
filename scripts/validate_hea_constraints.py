#!/usr/bin/env python3
"""Validate HEA compositions against constraint library."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONSTRAINT_PATH = BASE_DIR / "data" / "metadata" / "hea_constraints.yaml"

pattern = re.compile(r"([A-Z][a-z]?)([0-9.]*?)")


@dataclass
class ValidationReport:
    total: int
    passed: int
    failed: int
    failures: List[Dict[str, object]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "failures": self.failures,
            },
            indent=2,
        )


def load_constraints() -> Dict[str, object]:
    with open(CONSTRAINT_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_formula(formula: str) -> Dict[str, float]:
    entries = pattern.findall(formula.replace(" ", ""))
    comps = {}
    for element, amount in entries:
        value = float(amount) if amount else 1.0
        comps[element] = comps.get(element, 0.0) + value
    total = sum(comps.values())
    if total == 0:
        return {}
    for element in comps:
        comps[element] /= total
    return comps


def validate_row(row: pd.Series, constraints: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    comp_rules = constraints["composition_rules"]
    allowed = set(comp_rules["allowed_elements"])
    min_unique = comp_rules["min_unique_elements"]
    max_unique = comp_rules["max_unique_elements"]
    min_fraction = comp_rules["min_atomic_fraction"]
    max_fraction = comp_rules["max_atomic_fraction"]

    composition = parse_formula(row["formula"])
    if not composition:
        issues.append("Invalid formula parsing")
    else:
        unique = len(composition)
        if unique < min_unique or unique > max_unique:
            issues.append(f"Unique elements {unique} outside [{min_unique}, {max_unique}]")
        if not all(elem in allowed for elem in composition):
            disallowed = [elem for elem in composition if elem not in allowed]
            issues.append(f"Disallowed elements: {disallowed}")
        for elem, fraction in composition.items():
            if fraction < min_fraction or fraction > max_fraction:
                issues.append(f"Element {elem} fraction {fraction:.2f} outside range")
                break

    phase_rules = constraints["phase_rules"]
    acceptable = set(phase_rules["acceptable_labels"])
    label = row.get("phase_label")
    if label not in acceptable:
        issues.append(f"Phase label {label} not acceptable")

    prop_bounds = constraints["property_bounds"]
    for column, bounds in prop_bounds.items():
        value = row.get(column)
        if pd.notna(value):
            if value < bounds["min"] or value > bounds["max"]:
                issues.append(f"{column} value {value} out of bounds {bounds}")

    quality = constraints["quality_checks"]
    if quality.get("require_density_pair"):
        exp_density = row.get("exp_density_g_cm3")
        calc_density = row.get("calc_density_g_cm3")
        if pd.notna(exp_density) and pd.isna(calc_density):
            issues.append("Calculated density missing while experimental density present")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HEA dataset against constraint library")
    parser.add_argument("input", help="Path to HEA dataset (Parquet or CSV)")
    parser.add_argument("--output", help="Optional path to write JSON report")
    args = parser.parse_args()

    constraints = load_constraints()
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    failures: List[Dict[str, object]] = []
    passed = 0
    for _, row in df.iterrows():
        issues = validate_row(row, constraints)
        if issues:
            failures.append({"reference_id": row.get("reference_id"), "formula": row.get("formula"), "issues": issues})
        else:
            passed += 1

    report = ValidationReport(total=len(df), passed=passed, failed=len(failures), failures=failures)
    if args.output:
        Path(args.output).write_text(report.to_json(), encoding="utf-8")
    else:
        print(report.to_json())


if __name__ == "__main__":
    main()
