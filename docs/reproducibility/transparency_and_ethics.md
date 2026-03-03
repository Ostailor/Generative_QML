# Transparency and Ethics Statement (`T8.3`)

## Scope and Evidence Policy
This project reports headline outcomes using production DFT and hardware-adapter evidence.
Simulation-only artefacts are retained for ablation context and are explicitly labeled.

## Data and Model Transparency
1. Data lineage is documented from raw sources through processed features and release manifests.
2. Model outputs are linked to MLflow runs and reproducible script entry points.
3. DFT outputs are versioned with checksums and campaign-level provenance metadata.

## Bias and Limitations
1. Candidate quality depends on source dataset coverage and descriptor quality.
2. Hardware-adapter pilots approximate backend behavior and should not be interpreted as physical synthesis validation.
3. Real DFT campaign reflects screened candidates, not exhaustive search of composition space.

## Responsible Use
1. Downstream experimental work should independently validate top candidates.
2. Safety and environmental implications of target alloy chemistries should be assessed before synthesis.
3. Report both positive and negative outcomes in follow-on studies to avoid survivorship bias.

## Compliance Notes
- Reproducibility artefacts and manifests are included in release bundles.
- Submission package includes data availability and methods reproducibility sections.
