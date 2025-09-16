# Dataset Release v1.0

## Contents
This release bundles curated raw and processed datasets supporting the Generative Quantum Machine Learning project:

- Raw data sources
  - `data/raw/perovskites/materials_project_perovskites.csv`
  - `data/raw/high_entropy_alloys/High Entropy Alloy Properties.csv`
  - `data/raw/doped_nanoparticles/catalysis_hub_single_atom_alloy.csv`
- Processed feature tables
  - `data/processed/perovskites_features.parquet`
  - `data/processed/hea_features.parquet`
  - `data/processed/saa_features.parquet`
- Metadata & QA
  - `data/metadata/data_catalog.md`
  - `data/metadata/provenance_manifest.csv`
  - `data/metadata/qa_reports/*.json`
  - `data/metadata/hea_constraints.yaml`
  - `data/metadata/qa_reports/hea_constraints_validation.json`
  - `data/metadata/qa_reports/noise_simulations_summary.json`
  - `data/metadata/qa_reports/dft_handoff_validation.json`
- DFT handoff sample packages (`data/dft_handoff/input/QAL-0001`, `data/dft_handoff/output/QAL-0001`).

## Reproduction Steps
1. Create and activate a Python 3.11 environment.
2. Install preprocessing dependencies:
   ```
   pip install -r requirements-preprocess.txt
   pip install -r requirements-noise.txt
   ```
3. Rebuild processed datasets:
   ```
   python scripts/preprocess_datasets.py
   ```
4. Generate noise simulations:
   ```
   python scripts/simulate_noise.py
   ```
5. Validate HEA constraints and DFT handoff packages:
   ```
   python scripts/validate_hea_constraints.py data/processed/hea_features.parquet --output data/metadata/qa_reports/hea_constraints_validation.json
   python scripts/validate_dft_handoff.py
   ```

All outputs should match the checksums recorded in `release_manifest.json`.

## Provenance
- Raw data sources detailed in `data/metadata/data_catalog.md`.
- DFT interface described in `docs/interfaces/dft_handoff.md`.

