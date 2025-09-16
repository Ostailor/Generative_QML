#!/usr/bin/env python3
"""Generate workspace status snapshot from MLflow acceptance registry."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

try:
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Run
except ImportError as exc:  # pragma: no cover - informative exit for missing dependency
    sys.stderr.write(
        "[ERROR] mlflow is required for tracking. Install with `pip install mlflow`\n"
    )
    raise


SAFE_GLOBALS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "math": math,
    "len": len,
    "bool": bool,
    "all": all,
    "any": any,
}


@dataclass
class AcceptanceRule:
    milestone_id: str
    task_id: str
    description: str
    metric: str
    tag_filter: str
    template: str
    expression: str
    frequency: str

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "AcceptanceRule":
        return cls(
            milestone_id=row["milestone_id"],
            task_id=row["task_id"],
            description=row["acceptance_criterion"],
            metric=row["mlflow_metric"],
            tag_filter=row["mlflow_tag"],
            template=row["workspace_template"],
            expression=row["target_expression"],
            frequency=row["monitoring_frequency"],
        )


@dataclass
class RuleStatus:
    rule: AcceptanceRule
    status: str  # pass / fail / missing
    latest_value: Optional[str]
    run_id: Optional[str]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="tracking/acceptance_criteria_registry.csv",
        help="Path to acceptance registry CSV",
    )
    parser.add_argument(
        "--output",
        default="workspace/registers/status_snapshot.md",
        help="Output markdown file for snapshot",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--week-ending",
        default=dt.date.today().isoformat(),
        help="ISO date to label the report",
    )
    return parser.parse_args()


def load_registry(path: str) -> List[AcceptanceRule]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [AcceptanceRule.from_row(row) for row in reader]


def fetch_latest_run(client: MlflowClient, tag_filter: str) -> Optional[Run]:
    if not tag_filter:
        return None
    if ":" not in tag_filter:
        raise ValueError(f"Tag filter '{tag_filter}' must be in key:value format")
    key, value = tag_filter.split(":", 1)
    experiments = client.search_experiments()
    latest_run: Optional[Run] = None
    for exp in experiments:
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.`{key}` = '{value}'",
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        if runs:
            run = runs[0]
            if latest_run is None or run.info.start_time > latest_run.info.start_time:
                latest_run = run
    return latest_run


def assign_nested(env: Dict[str, object], dotted_name: str, value: object) -> None:
    if not dotted_name:
        return
    parts = dotted_name.split(".")
    cursor = env
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
    cursor[parts[-1]] = value


def build_env(run: Run) -> Dict[str, object]:
    env: Dict[str, object] = {}
    metrics = run.data.metrics
    params = run.data.params
    tags = run.data.tags
    for name, value in metrics.items():
        assign_nested(env, name, value)
    for name, value in params.items():
        assign_nested(env, name, value)
    for name, value in tags.items():
        assign_nested(env, name, value)
    # Provide convenience alias for workspace validation tag
    if "workspace_validation" not in env and "workspace" in env:
        maybe = env["workspace"]
        if isinstance(maybe, dict) and "validation" in maybe:
            env["workspace_validation"] = maybe["validation"]
    return env


def to_namespace(obj: object) -> object:
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: to_namespace(val) for key, val in obj.items()})
    return obj


def resolve_metric(env: Dict[str, object], dotted_name: str) -> Optional[object]:
    if not dotted_name:
        return None
    cursor: object = env
    for part in dotted_name.split("."):
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        else:
            return None
    return cursor


def evaluate_rule(rule: AcceptanceRule, run: Optional[Run]) -> RuleStatus:
    if run is None:
        return RuleStatus(rule, "missing", None, None, "No MLflow run found for tag")
    env_dict = build_env(run)
    namespace = to_namespace(env_dict)
    local_env = {**SAFE_GLOBALS, **env_dict}
    # allow attribute access
    local_env.update(namespace.__dict__ if hasattr(namespace, "__dict__") else {})
    try:
        result = eval(rule.expression, {"__builtins__": {}}, local_env)
        status = "pass" if bool(result) else "fail"
    except Exception as exc:  # pragma: no cover - evaluation error path
        return RuleStatus(
            rule,
            "fail",
            latest_value=str(env_dict.get(rule.metric, "n/a")),
            run_id=run.info.run_id,
            notes=f"Evaluation error: {exc}",
        )
    latest_val = resolve_metric(env_dict, rule.metric) if rule.metric else None
    if isinstance(latest_val, float):
        latest_repr = f"{latest_val:.5f}"
    else:
        latest_repr = str(latest_val)
    return RuleStatus(
        rule,
        status,
        latest_repr,
        run.info.run_id,
        "",
    )


def format_statuses(statuses: Iterable[RuleStatus], week_ending: str) -> str:
    lines = ["# Weekly Status Snapshot", "", f"- **Week Ending**: {week_ending}"]
    lines.append("- **Report Generated**: " + dt.datetime.utcnow().isoformat() + "Z")
    lines.append("- **Data Source**: MLflow + workspace registers")
    lines.append("")
    lines.append("## Acceptance Coverage")
    lines.append("| Milestone | Task | Status | Latest Value | MLflow Run | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for status in statuses:
        run_link = status.run_id or "n/a"
        latest = status.latest_value or "n/a"
        note = status.notes or ""
        lines.append(
            f"| {status.rule.milestone_id} | {status.rule.task_id} | {status.status} | {latest} | {run_link} | {note} |"
        )
    lines.append("")
    lines.append("## Outstanding Items")
    lines.append("- Review failed or missing entries and schedule remediation with owning agents.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rules = load_registry(args.registry)
    client = MlflowClient(tracking_uri=args.mlflow_tracking_uri)
    statuses: List[RuleStatus] = []
    for rule in rules:
        run = fetch_latest_run(client, rule.tag_filter)
        status = evaluate_rule(rule, run)
        statuses.append(status)
    output_text = format_statuses(statuses, args.week_ending)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(output_text)
    print(f"Status snapshot written to {args.output}")


if __name__ == "__main__":
    main()
