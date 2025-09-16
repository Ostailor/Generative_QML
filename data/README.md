# Data Workspace Overview

This directory hosts datasets supporting M1 tasks for the quantum-active-learning research program. Each subdirectory is versioned and accompanied by provenance metadata to ensure reproducibility.

## Structure
- `raw/` — Source datasets as obtained (immutable).
  - `perovskites/`
  - `high_entropy_alloys/`
  - `doped_nanoparticles/`
- `metadata/` — Schema definitions, provenance manifests, and QA reports.
- `processed/` (to be created in M1 T1.2) — Feature-engineered outputs.

## Versioning
- Dataset release identifiers follow `{domain}-v{MAJOR.MINOR.PATCH}`.
- Provenance manifests include DOI/URL, license, checksum, acquisition date, and preprocessing status.

## Governance
- DPQA owns data ingestion; RKMA verifies provenance records.
- Raw data is read-only once recorded. Changes require new version directories and manifest updates.

