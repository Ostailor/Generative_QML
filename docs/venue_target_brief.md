# Venue Target Brief — Quantum Materials Dissemination Plan

## Primary Target (Conference)
- **Venue**: IEEE Quantum Computing and Engineering Conference (QCE) — Quantum Materials & Chemistry Track
- **Submission Deadline**: 2026-01-10 (Full Paper)
- **Format Requirements**:
  - 6-page limit (IEEE two-column template).
  - Abstract ≤ 250 words; up to 5 keywords.
  - Mandatory graphical abstract highlighting QAL pipeline architecture.
- **Data & Reproducibility Policies**:
  - Encourages open-source artefacts; provide GitHub/Zenodo link.
  - Requires disclosure of hardware providers and calibration procedures.
  - Recommends reproducibility checklist covering data availability, code, and computational resources.
- **Rationale**: Offers dedicated focus on quantum-enabled materials discovery and attracts both quantum computing and materials science audiences aligned with project objectives.

## Secondary Target (Conference)
- **Venue**: Materials Research Society (MRS) Spring Meeting — Symposium EQ06: Quantum Materials Discovery and AI
- **Submission Deadline**: 2025-12-05 (Proceedings Paper) / 2025-10-15 (Abstract)
- **Format Requirements**:
  - 8-page limit (MRS Word/LaTeX template).
  - Includes impact statement and detailed methods section.
  - Figures must be in CMYK, 300 dpi minimum.
- **Data & Reproducibility Policies**:
  - Requires data availability statement; encourages FAIR-compliant repositories.
  - Expects supplementary material for computational workflows (DFT scripts, ML configs).
- **Rationale**: Strong materials science community, emphasizes integration of AI and quantum techniques in materials development, provides networking with experimentalists.

## Journal Contingency
- **Venue**: npj Quantum Materials (Journal)
- **Submission Window**: Rolling; aim for 2026-Q1 submission if conferences decline.
- **Format Requirements**:
  - 6,000-word limit (excluding methods and references).
  - Unlimited figures/tables; short methods summary with extended supplementary info allowed.
  - Requires structured abstract (Background, Results, Conclusions).
- **Data & Reproducibility Policies**:
  - Mandates data availability statement and deposition of code/data in public repositories.
  - Encourages providing raw and processed DFT data with metadata.
  - Peer review expects detailed calibration notes for QPU experiments.
- **Rationale**: High-impact open-access journal covering quantum materials and computation; suitable for extended results including robustness studies.

## Cross-Venue Compliance Checklist
1. Maintain dual-format manuscript sources (IEEE + MRS + Nature template variations).
2. Reserve MLflow experiment export and provenance snapshot for supplemental material.
3. Ensure DFT datasets are deposited with DOIs before submission.
4. Prepare hardware calibration appendix with cost/fidelity tables.
5. Schedule internal review timeline backward from earliest deadline (2025-10-15 abstract) to lock milestones M7–M9.

## Action Items
- PDA to confirm final venue choice by 2025-10-01 after assessing milestone progress.
- RKMA to assemble template bundle under `docs/templates/`.
- BRA to tailor evaluation plots to meet each venue’s figure requirements.
- QHSOA to prepare hardware compliance statement for submission packages.

