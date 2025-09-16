# M0 Research Charter — Generative Quantum Machine Learning for Cost-Efficient, High-Strength Materials Discovery

## 1. Purpose
Establish the scope, success metrics, governance, and initial risk posture for the research program exploring quantum-active-learning (QAL) pipelines that integrate QSVR/QGPR models, quantum generative circuits, and DFT feedback for accelerated discovery of high-strength, cost-efficient materials.

## 2. Strategic Objectives
1. **Modular QAL Pipeline**: Deliver a closed-loop workflow that couples quantum regressors, quantum generative models, Bayesian acquisition strategies, and automated DFT evaluation.
2. **Benchmarking & Efficiency**: Quantify predictive accuracy, label-efficiency gains (target ≥30% reduction in DFT calls), robustness to noise, and hardware performance vs simulators.
3. **Generative Discovery**: Produce property-conditional candidate HEA compositions satisfying mechanical and thermodynamic targets validated via DFT.
4. **Reproducible Publication**: Submit a research-grade manuscript to a leading quantum materials venue with open artefact packages.

## 3. Scope
### In-Scope
- Quantum kernel regression (QSVR/QGPR) design, implementation, and benchmarking.
- Quantum generative model development for HEA composition proposals.
- Active learning loop orchestration integrating QPU feedback and DFT workflows.
- Dataset curation (perovskites, HEAs, doped nanoparticles) with quality controls.
- Hybrid simulator/QPU experimentation with cost and fidelity tracking.
- Statistical benchmarking against classical active learning baselines.
- Reproducibility infrastructure (workspace, experiment logging, provenance graphs).
- Manuscript drafting, internal review, and submission management.

### Out of Scope / Deferred
- Experimental synthesis or physical validation beyond DFT simulations.
- Quantum hardware development; only usage and calibration of existing backends.
- Expansion beyond target material classes unless approved in milestone reviews.

## 4. Success Metrics
| Dimension | Metric | Target |
| --- | --- | --- |
| Label Efficiency | Reduction in DFT evaluations vs classical AL | ≥30% average reduction (with statistical significance p<0.05) |
| Predictive Accuracy | RMSE/MAE of QSVR/QGPR vs classical baselines | ≤10% degradation or demonstrable trade-off justification |
| Generative Validity | Fraction of samples satisfying HEA constraints | ≥85% acceptance rate post-filter |
| Novelty | Statistical improvement in novelty/feasibility balance | p<0.05 vs classical generators |
| Hardware Performance | Simulator vs QPU metric deviation | Within agreed tolerance; documented cost per run |
| Reproducibility | Successful rerun of full pipeline in clean environment | 100% automated reproduction |
| Publication | Conference/journal submission | Manuscript submitted before venue deadline |

## 5. Milestones & Gate Criteria
| Milestone | Summary | Gate Criteria |
| --- | --- | --- |
| M0 | Program alignment & literature reconnaissance | Charter approved, literature synthesis ≥40 sources, venue brief recorded, workspace validated |
| M1 | Data readiness & domain constraints | Dataset v1.0 released with QA report; HEA constraint library approved |
| M2 | Quantum kernel regression foundations | QSVR/QGPR prototypes meeting accuracy targets; classical baselines established |
| M3 | Quantum generative modeling capability | Property-conditional models achieving ≥85% valid samples; novelty improvement significant |
| M4 | Active learning loop design | Closed-loop simulator workflow meeting label-efficiency target |
| M5 | DFT integration & feedback | Automated DFT feedback integrated; performance gains documented |
| M6 | Hardware execution & cost management | QPU runs completed with mitigation plans and cost dashboards |
| M7 | Benchmarking & robustness | Robustness sweeps + hardware comparison reports approved |
| M8 | Reproducibility & packaging | Provenance graph, containerized artefacts, compliance checklist complete |
| M9 | Manuscript authoring & submission | Internal reviews cleared; submission package archived |

Gate reviews are chaired by PDA with participation from relevant agent leads. Each gate requires:
- Updated risk register with mitigations in place.
- Acceptance criteria evidence logged in MLflow and workspace templates.
- Decision log entry capturing go/no-go outcomes.

## 6. Governance & Cadence
- **Steering Cadence**: Bi-weekly program sync (PDA + agent leads); milestone gate reviews aligned with roadmap.
- **Decision Authority**: PDA approves scope changes, resource allocation, and milestone exits. Technical leads own implementation decisions within their domain.
- **Documentation**: All significant decisions and experiment logs recorded using workspace templates adhering to `metadata_schema.yaml`.
- **Experiment Tracking**: MLflow server (default `http://localhost:5001`) storing metrics tied to acceptance registry; weekly status snapshot generated via `tracking/reporting/update_status_snapshot.py`.

## 7. Resources & Tooling Assumptions
- **Personnel**: Agents defined in `AGENTS.md` with PDA acting as project owner.
- **Budget**: QPU credit allocation capped at baseline IonQ/Quantinuum plan; expansion requires PDA approval.
- **Toolchain**: Python 3.11 environment with MLflow ≥2.9, quantum SDKs (PennyLane/Qiskit), DFT toolkit (VASP/Quantum ESPRESSO via automation scripts), CI for data pipelines, documentation workspace as configured in M0.
- **Infrastructure**: Local and cloud compute for simulators, secure storage for datasets and DFT outputs, repository-based version control.

## 8. Initial Risk Register
| ID | Description | Impact | Likelihood | Owner | Mitigation |
| --- | --- | --- | --- | --- | --- |
| R1 | QPU access throttling delays hardware experiments | Schedule | Medium | PDA | Secure reservation slots; maintain simulator fallback; prioritize critical circuits |
| R2 | DFT workflow bottlenecks due to compute resource limits | Schedule/Quality | Medium | MDIA | Establish priority queue; leverage HPC credits; optimize workflow automation |
| R3 | Quantum kernels fail to match classical baseline accuracy | Quality | Medium | QKAA | Expand feature-map search; hybridize with classical kernels; early ablation studies |
| R4 | Generative models produce chemically invalid HEA candidates | Quality | High | QGMA | Tighten MDIA constraints; incorporate rejection sampling & curriculum training |
| R5 | Reproducibility gaps from undocumented experiments | Compliance | Low | RKMA | Enforce template usage; weekly audits; integrate MLflow logging hooks |
| R6 | Conference deadline shifts or submission requirements change | Schedule | Low | PDA | Monitor CFP updates; maintain buffer in manuscript schedule |

## 9. Approvals
- **Prepared by**: PDA (Program Director Agent)
- **Reviewed by**: RKMA, QKAA, QGMA, ALOA, MDIA, DPQA, QHSOA, BRA
- **Approval Date**: _TBD_

Sign-off requires acknowledgement from all agent leads recorded via workspace decision log entry referencing this charter.

