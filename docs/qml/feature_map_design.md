# Feature Map Design Brief — M2 T2.1

## Objectives
- Tailor quantum feature maps to materials descriptors (composition vectors, local environments, HEA phase indicators).
- Provide parameter sweep plan and expressivity heuristics guiding QSVR/QGPR development.
- Document hardware resource estimates for candidate trapped-ion and superconducting backends (in coordination with QHSOA).

## Feature Map Families

### 1. HEA Composition Encoding (Fermionic-style Data Re-uploading)
- **Qubits**: 6 (supports up to 6 unique elemental fractions via one-hot + amplitude encoding).
- **Structure**: Data re-uploading layers with Ry rotations parameterized by normalized elemental fractions, interleaved with ZZ entanglers.
- **Parameters**: 2 * number_of_layers (θ, φ per qubit) + entangler weights.
- **Expressivity notes**: Capable of approximating polynomial interactions up to degree equal to number_of_layers; increasing layers beyond 3 shows diminishing returns per numerical expressivity metric (see table below).

### 2. Local Environment Embedding (Hardware-Efficient Entangling Ansatz)
- **Qubits**: 8 (encodes local coordination numbers + phase indicators).
- **Structure**: Alternating layers of Ry-Rz rotations with CZ ladder entanglement; parameterized by structural descriptors (bond angles, coordination numbers) from preprocessing pipeline.
- **Parameters**: 3 * qubits * layers.
- **Expressivity notes**: Hardware-efficient ansatz provides balanced trainability; barren plateau risk mitigated by layer ≤4.

### 3. Phase-Aware Kernel Map (Feature-Tensor Product)
- **Qubits**: 4 (binary phase indicators plus aggregated mechanical property bins).
- **Structure**: Composite map combining classical RBF kernel on mechanical descriptors with quantum phase oracle; implemented as controlled rotations conditioned on phase label (BCC/FCC/other).
- **Parameters**: Minimal (phase-controlled rotations only); uses tunable scale parameter γ for RBF component.
- **Expressivity notes**: Focused on low-parameter regime to avoid overfitting; suitable for resource-constrained QPUs.

## Parameter Sweep Plan
| Family | Layers Tested | Hyperparameters | Objective |
| --- | --- | --- | --- |
| Composition Encoding | 1–5 | {θ, φ} rotation angles per layer | Identify depth vs expressivity trade-off |
| Local Environment Ansatz | 2–4 | rotation scales, entangler weights | Balance expressivity and hardware runtime |
| Phase-Aware Map | N/A | RBF γ values {0.1, 0.5, 1.0} | Tune kernel sharpness for phase separation |

## Expressivity Heuristics
- Numerical expressivity proxy computed via average pairwise effective dimension (see `data/qml/feature_map_expressivity.csv`).
- Hardware depth estimates derived from Qiskit transpilation for target devices (IonQ Harmony, IBM Perth).

## Next Steps
- Implement feature maps in codebase (`qml/feature_maps.py`) using parameter templates below.
- Coordinate with QHSOA for calibration and mitigation strategy per backend.

