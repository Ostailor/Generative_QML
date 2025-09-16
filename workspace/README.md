# Workspace Overview

This workspace implements Task T0.4 in `TASKS.md` by defining a shared documentation environment where all agents record charters, decision logs, lab notebooks, and milestone reviews. The structure below aligns to the metadata schema and acceptance criteria required for continuous monitoring of research progress.

## Directory Layout
- `workspace/templates/` — Markdown templates for decision logs, lab notebook entries, milestone reviews, and risk updates.
- `workspace/metadata_schema.yaml` — Canonical metadata fields enforced across all documentation artefacts (IDs, agents, datasets, QPU backends, provenance hashes, etc.).
- `workspace/registers/` — Rolling CSV/Markdown registers capturing charter approvals, risk statuses, and completed milestones.
- `tracking/` — Experiment logging infrastructure (MLflow configuration, acceptance registry).

## Usage Principles
1. **Single Source of Truth**: All agents log decisions and experimental summaries in this workspace before tasks are considered closed.
2. **Metadata Validation**: Each entry must include the identifiers and fields specified in `workspace/metadata_schema.yaml`. RKMA audits compliance weekly.
3. **Linked Artefacts**: Reference hashed datasets, commit SHAs, and experiment run IDs from the experiment logger to maintain traceability.
4. **Version Control Discipline**: Treat this workspace as immutable history—append new entries rather than editing past decisions without an addendum.

## Onboarding Checklist (T0.4 Acceptance)
- [ ] Agent has write access to repository and workspace directories.
- [ ] Agent has reviewed metadata schema and templates.
- [ ] Agent submitted a sample entry using `templates/lab_notebook_entry.md` populated with mock data.
- [ ] PDA recorded acknowledgement in `registers/onboarding_log.csv`.

## Integration with Experiment Logger
- Every experiment run tracked in MLflow must reference its corresponding task ID and acceptance criterion label (see `tracking/acceptance_criteria_registry.csv`).
- The `experiment_summary.md` template includes a placeholder for MLflow Run URI to link the documentation entry with logger artefacts.

## Governance
- PDA owns approvals of milestone reviews; RKMA owns template updates and metadata schema changes.
- Updates to schema or templates require pull request review by PDA + RKMA.
- Weekly sync uses `registers/status_snapshot.md` generated from experiment logger metrics to review acceptance coverage.

