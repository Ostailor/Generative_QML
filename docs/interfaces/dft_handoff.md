# DFT Handoff Interface Specification (T1.5)

This document defines the contract between data/preprocessing agents (DPQA) and materials/DFT integration agent (MDIA) for exchanging inputs and outputs used in closed-loop active learning.

## 1. Input Package Format (DPQA → MDIA)
- **Location**: `data/dft_handoff/input/<request_id>/`
- **Required files**:
  - `metadata.json`: JSON document containing request metadata:
    - `request_id` (string, unique)
    - `source_dataset` (string, e.g., `hea_features`)
    - `timestamp_utc` (ISO-8601)
    - `composition` (string, normalized formula)
    - `constraints_version` (string referencing `hea_constraints.yaml`)
    - `target_properties` (array of strings requested, e.g., `yield_strength_mpa`)
    - `notes` (string, optional)
  - `structure.cif`: Crystal structure in CIF format (for HEA approximations generated via Vegard-like models).
  - `vasp_settings.json`: Encodes DFT calculator inputs:
    - `functional`: e.g., `PBE`
    - `encut`: planewave cutoff (eV)
    - `kpoint_grid`: `[kx, ky, kz]`
    - `ediff`: electronic convergence tolerance
    - `smearing`: scheme + width
    - `spin_polarized`: boolean
  - `pseudopotentials.csv`: CSV table with columns `element`, `potential`, `library`.
- **Optional files**:
  - `initial_magnetic_moments.json`
  - `pre_relaxation_summary.json`

## 2. Output Package Format (MDIA → DPQA)
- **Location**: `data/dft_handoff/output/<request_id>/`
- **Required files**:
  - `results.json`: JSON with keys:
    - `request_id`
    - `status` (`completed`, `failed`, `in_progress`)
    - `total_energy_eV`
    - `formation_energy_eV`
    - `properties`: object containing computed property values (e.g., `bulk_modulus_GPa`, `yield_strength_mpa_est`, etc.)
    - `uncertainty`: object with standard deviations for each property
    - `dft_settings_hash`: SHA256 of input package to guarantee provenance
  - `structure_relaxed.cif`: relaxed structure.
  - `log.txt`: key DFT workflow logs (SCF convergence, warnings).
- **Optional files**:
  - `phonon_summary.json`, `elastic_constants.json`
  - `error_report.json` (if `status = failed`)

## 3. Naming and Versioning
- Use UUID or AL iteration ID for `request_id` (e.g., `qal-iter05-sample001`).
- Version control interface via `data/metadata/hea_constraints.yaml` and document updates in this file; increment `constraints_version` when schema changes.

## 4. Validation
- DPQA maintains `scripts/validate_dft_handoff.py` to check packages:
  - Ensures required files exist.
  - Validates JSON schema against `metadata_schema.yaml` subset.
  - Verifies `dft_settings_hash` matches SHA256 of serialized input files.
- Validation outputs stored under `data/metadata/qa_reports/dft_handoff_validation.json`.

## 5. Workflow Summary
1. DPQA generates candidate, prepares input package under `data/dft_handoff/input/<request_id>/`.
2. Validation script runs; package zipped and delivered to MDIA workflow.
3. MDIA runs DFT, writes outputs to corresponding output directory, updates `results.json`.
4. Validation script verifies outputs (hash, status, presence of relaxed structure).
5. MLflow run logs metrics (`mdia.dft_requests_completed`, etc.) for traceability.

