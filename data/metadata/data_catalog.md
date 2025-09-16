# Data Catalog — M1 T1.1

## Perovskite Property Dataset
- **Proposed Source**: Materials Project API (Kubota *et al.*, 2020 dataset ID mp-perovskites); supplementary CSV hosted on Figshare.
- **Primary DOI**: 10.6084/m9.figshare.11890311
- **License**: CC-BY 4.0
- **Content Summary**: ~6,000 compositions with bandgap, formation energy, tolerance factors; includes crystal structures in CIF format.
- **Schema Snapshot**:
  - `material_id` (string)
  - `composition` (string; normalized chemical formula)
  - `band_gap_eV` (float)
  - `formation_energy_per_atom` (float)
  - `stability_energy_above_hull` (float)
  - `spacegroup` (string)
  - `cif_path` (string)
- **Ingestion Plan**: Use Materials Project REST API (pymatgen) with API key; store JSON dump and normalized CSV in `raw/perovskites`.

## High-Entropy Alloy Dataset
- **Proposed Source**: NIMS HEA database (Miracle & Senkov, 2017) via Kaggle repository `high-entropy-alloys.csv`.
- **Primary DOI**: 10.24435/materialscloud:2017.0002/v1
- **License**: CC-BY-SA 4.0
- **Content Summary**: 1,500 alloy compositions with mechanical properties (yield strength, hardness), phase data, processing conditions.
- **Schema Snapshot**:
  - `alloy_id` (string)
  - `elements` (list[str])
  - `processing_route` (string)
  - `phase` (categorical)
  - `yield_strength_MPa` (float)
  - `hardness_HV` (float)
  - `density_g_cm3` (float)
- **Ingestion Plan**: Download CSV + supplemental JSON for processing metadata; store raw assets and license notes in `raw/high_entropy_alloys`.

## Doped Nanoparticle Dataset
- **Source**: Catalysis-Hub.org collection — Single Atom Alloy (SAA) adsorption energetics.
- **Primary DOI**: 10.24435/materialscloud:2020.0046/v3 (Catalysis-Hub SAA dataset)
- **License**: CC-BY 4.0
- **Content Summary**: ≈1,000 entries describing single-atom alloy (doped nanoparticle) compositions, adsorption sites, reaction energies, and metadata.
- **Schema Snapshot**:
  - `reaction_id` (string)
  - `chemical_composition` (string)
  - `facet` (string)
  - `site` (string)
  - `adsorbate` (string)
  - `reaction_energy_eV` (float)
  - `source` (string)
- **Ingestion Plan**: Query Catalysis-Hub GraphQL/REST endpoint with limit configurable via environment variable; store JSON + CSV exports and record query metadata.

## Shared QA Requirements
- Verify column presence and datatypes using `scripts/ingest_datasets.py` validation routines.
- Compute row counts and missingness metrics; thresholds logged to MLflow (target completeness ≥99%).
- Record SHA256 checksums in `provenance_manifest.csv` after downloads.
