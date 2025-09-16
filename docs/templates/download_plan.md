# Template Acquisition Plan

## Responsibilities
- **Owner**: RKMA
- **Support**: PDA (for license approvals), BRA (figure requirements), QHSOA (hardware appendix inputs).

## Download Checklist
1. **IEEE QCE Templates**
   - Access IEEE Author Center and download latest LaTeX (`IEEEtran.zip`) and Word templates.
   - Replace placeholders `ieee_conf_template.tex` and `.docx` in `docs/templates/IEEE_QCE/`.
   - Store PDF eXpress compliance report.
2. **MRS Spring Meeting Templates**
   - Retrieve proceedings template package from MRS submission portal (LaTeX and Word).
   - Obtain official CMYK ICC profile; overwrite `color_profile.icc` placeholder.
   - Export impact statement examples for reference.
3. **npj Quantum Materials Templates**
   - Download Nature Master Template and structured abstract guidelines.
   - Add data availability form and licensing instructions.

## Version Control Protocol
- Commit template updates with version tags: `templates-IEEE-QCE-<YYYYMMDD>`, etc.
- Include provenance metadata (download URL, date, checksum) in each subdirectory README.
- Store binary templates (Word, PDF) using Git LFS if size > 100 MB; otherwise track directly.

## Coordination Timeline
- Deadline: 2025-09-25 (per T0.3 action items).
- Status Updates: Provide progress notes during weekly PDA sync; log completion via workspace decision log.

