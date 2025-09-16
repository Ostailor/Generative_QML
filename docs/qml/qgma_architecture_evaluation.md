# Quantum Generative Architecture Evaluation — M3 T3.1

## Candidate Architectures
| Architecture | Variants | Expressivity Score | Resource Notes |
| --- | --- | --- | --- |
| QVAE | 2 | 0.72 | 6 qubits, depth ~30, requires amplitude encoding |
| QGAN | 3 | 0.81 | 8 qubits, depth ~40, moderate training stability |
| Data re-uploading VQC | 2 | 0.75 | 6 qubits, depth ~25, aligns with composition feature maps |

## Selection Criteria
- Expressivity vs training stability
- Hardware resource demands (qubits, gate depth)
- Integration compatibility with HEA feature encodings

## Recommended Path
- Primary: QGAN (highest expressivity; proceed with stability mitigation).
- Secondary: Data re-uploading VQC (fallback with lower depth).

## Next Steps
- Implement QGAN prototype (T3.2) with HEA constraints integration.
- Use VQC as backup for ablation studies.

