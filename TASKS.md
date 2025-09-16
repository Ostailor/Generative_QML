# TASKS.md

## Document Purpose
Map the end-to-end work program required to deliver a conference-ready manuscript on "Generative Quantum Machine Learning for Cost-Efficient, High-Strength Materials Discovery". Tasks are organized by milestone, aligned to the agent architecture in `AGENTS.md`, and include dependencies, deliverables, and acceptance criteria to guarantee research-grade rigor and reproducibility.

## Milestone Roadmap
| Milestone ID | Theme | Target Timeline* | Exit Criteria |
| --- | --- | --- | --- |
| M0 | Program alignment & literature reconnaissance | Weeks 1-2 | Charter, literature synthesis, and conference target brief approved by PDA |
| M1 | Data readiness & domain constraints | Weeks 2-5 | Versioned datasets with QA reports and HEA constraint library released |
| M2 | Quantum kernel regression foundations | Weeks 4-8 | QSVR/QGPR prototypes with validated feature maps and baseline metrics |
| M3 | Quantum generative modeling capability | Weeks 6-10 | Property-conditional generative models producing chemically valid HEA candidates |
| M4 | Active learning loop design | Weeks 9-12 | Closed-loop simulation orchestrations validated on simulators |
| M5 | DFT integration & feedback coupling | Weeks 11-15 | Automated DFT workflows providing uncertainty-aware feedback |
| M6 | Hardware execution & cost management | Weeks 13-18 | QPU-calibrated circuits with cost/fidelity dashboards |
| M7 | Benchmarking & robustness evaluation | Weeks 16-20 | Comparative results versus classical AL baselines with robustness analyses |
| M8 | Reproducibility, packaging & release prep | Weeks 18-21 | Reproducible artifact registry and final experiment provenance reports |
| M9 | Manuscript authoring & submission | Weeks 20-24 | Full paper draft, internal reviews, and submission package completed |

*Timelines are indicative for sequencing; adjust dynamically via PDA governance.

---

## M0 — Program Alignment & Literature Reconnaissance
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T0.1 | PDA | Formalize research charter, scope boundaries, risk register, and quarterly decision cadence. | Project brief, AGENTS.md | Charter doc, risk register, milestone gate checklist. | PDA and agent leads sign-off; risks categorized with mitigation owners. | None |
| T0.2 | RKMA (w/ PDA) | Conduct structured literature review across QAL, QSVR/QGPR, HEA discovery, and quantum generative models. Build annotated bibliography. | Databases, prior notes | Zotero/Notion library export, synthesis memo summarizing research gaps. | Minimum 40 peer-reviewed sources tagged; memo highlights at least 5 actionable insights per aim. | T0.1 |
| T0.3 | PDA | Select target Quantum Materials Science conferences/journals, capture submission guidelines, deadlines, and formatting. | Conference calls, T0.2 outputs | Venue target brief, template bundle (LaTeX/Word), compliance checklist. | Brief accepted by RKMA; guidelines stored in repo and accessible to all agents. | T0.2 |
| T0.4 | RKMA | Stand up shared documentation workspace (e.g., lab notebook structure, decision log schema) and align metadata standards. | T0.1 charter | Workspace SOP, metadata schema doc, onboarding guide. | All agents acknowledge access; metadata schema validated through sample entry. | T0.1 |

## M1 — Data Readiness & Domain Constraints
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T1.1 | DPQA | Aggregate perovskite, HEA, and doped nanoparticle datasets with provenance metadata; harmonize schemas. | Source repositories, T0.2 bibliography | Raw data catalog, ingestion scripts, provenance manifest. | Completeness and schema validation checks >99%; provenance includes source DOI/URL and license. | T0.4 |
| T1.2 | DPQA | Build preprocessing & feature engineering pipelines (composition embeddings, graph descriptors). | T1.1 dataset | Versioned preprocessing scripts, feature summary report. | CI pipeline passes unit/data validation tests; report quantifies feature distributions and missingness. | T1.1 |
| T1.3 | DPQA | Implement noise/perturbation simulation module for robustness experiments. | T1.2 pipeline | Noise injection library with parameter docs, synthetic datasets. | Module validated on holdout set; BRA approves coverage of noise scenarios. | T1.2 |
| T1.4 | MDIA | Develop HEA domain constraint library (phase stability heuristics, composition priors, property targets). | T1.1 data, literature | Constraint specification document, rule engine prototype. | Constraint checks cover ≥95% of historical HEA cases; PDA approves for integration. | T1.1 |
| T1.5 | MDIA & DPQA | Define DFT input formats and data handoff contracts to ensure compatibility with ACL loops. | T1.2 features, T1.4 constraints | Interface spec, sample data packages, validation notebook. | Successful round-trip test between data pipeline and mock DFT workflow. | T1.2, T1.4 |
| T1.6 | RKMA | Release Dataset v1.0 with hashed artefacts, README, and reproducibility checklist. | T1.1-T1.5 outputs | Data release package, DOI/internal registry entry. | BRA signs off on data readiness; package replicable via scripted command. | T1.5 |

## M2 — Quantum Kernel Regression Foundations
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T2.1 | QKAA | Design feature-map families and parameter sweeps tailored to materials descriptors; document hardware resource estimates. | T1.2 features, T1.4 constraints | Feature-map design brief, circuit schematics, resource table. | Includes expressivity metrics; QHSOA confirms hardware feasibility. | T1.4 |
| T2.2 | QKAA | Implement QSVR prototypes on simulators; evaluate against baseline classical kernels. | T2.1 designs, T1.6 data | Training notebooks, performance report (RMSE, MAE, calibration). | Performance vs classical <10% degradation on core metrics; reproducible seeds logged. | T2.1 |
| T2.3 | QKAA | Develop QGPR models with uncertainty estimates; explore parameterized quantum kernels. | T2.1 brief, T1.6 data | QGPR code, uncertainty calibration plots, comparative analysis. | Coverage probability within ±5% of nominal level; BRA endorses statistical validity. | T2.2 |
| T2.4 | BRA | Establish classical AL baseline suite (e.g., Gaussian process, RF AL) for reference. | T1.6 data, T2.2 metrics | Baseline benchmark report, code modules, evaluation scripts. | Benchmarks reproducible; results stored in evaluation registry with seeds. | T2.2 |
| T2.5 | RKMA | Document QSVR/QGPR experiments in lab notebook with provenance captured. | T2.2, T2.3 outputs | Notebook entries, provenance graph update. | Entries pass metadata schema validation; PDA signs off on completeness. | T2.3 |

## M3 — Quantum Generative Modeling Capability
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T3.1 | QGMA | Survey candidate property-conditional quantum generative architectures; down-select with scoring rubric. | Literature (T0.2), T1.4 constraints | Architecture evaluation matrix, selection memo. | At least 3 architectures scored; PDA approves selected path. | T1.4 |
| T3.2 | QGMA | Implement selected generative model(s) on simulators; integrate chemical validity filters. | T3.1 memo, T1.6 data | Generative model code, sampling diagnostics, invalid rejection log. | ≥85% of accepted samples satisfy MDIA constraints; log tracks failure categories. | T3.1, T1.6 |
| T3.3 | QGMA & MDIA | Incorporate property targets and DFT-informed priors into generative loss functions. | T1.5 handoff, T3.2 model | Updated model spec, loss function documentation, pilot results. | Demonstrated control of target property within ±10% on validation set. | T3.2, T1.5 |
| T3.4 | BRA | Evaluate generative diversity and novelty metrics vs classical generators (e.g., VAE, GAN). | T3.2 outputs | Comparative report, novelty score histograms. | Quantum model exhibits statistically significant (p<0.05) improvement in novelty/feasibility trade-off. | T3.2, T2.4 |
| T3.5 | RKMA | Archive generative experiments and candidate libraries with provenance tags. | T3.2-3.4 outputs | Candidate repository, provenance graph update. | Repository hashed and indexed; metadata meets FAIR standards. | T3.4 |

## M4 — Active Learning Loop Design
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T4.1 | ALOA | Design end-to-end QAL loop architecture integrating QSVR/QGPR, generative proposals, and acquisition functions. | T2.2-2.3, T3.2-3.3 outputs | System architecture doc, sequence diagrams, API contracts. | Reviewed by QKAA, QGMA, MDIA; interfaces approved. | T3.3 |
| T4.2 | ALOA | Implement acquisition strategies (e.g., expected improvement, max information gain) suited to high-dimensional HEA search. | T4.1 architecture, T1.6 data | Acquisition module code, tuning results. | Demonstrates convergence improvements vs random sampling in simulator tests. | T4.1 |
| T4.3 | ALOA & DPQA | Build active-learning orchestration pipeline with experiment tracking hooks (integration with RKMA). | T4.2 module, T0.4 workspace | Orchestration scripts, experiment tracker integration. | Pipeline runs automated loop on simulator; logs captured in RKMA system. | T4.2, T0.4 |
| T4.4 | QKAA & QGMA | Validate model update cadence within active loop, ensuring stability and convergence. | T4.3 pipeline | Validation report, updated hyperparameter schedules. | Loop achieves target label-efficiency (≥30% reduction) in simulated setting. | T4.3 |
| T4.5 | PDA | Gate review of QAL loop readiness for DFT integration. | T4.4 report | Decision memo, action items. | Go/No-Go decision documented; outstanding risks assigned owners. | T4.4 |

## M5 — DFT Integration & Feedback Coupling
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T5.1 | MDIA | Implement automated DFT workflow (structure generation, relaxation, property extraction) with uncertainty estimates. | T1.5 interface, T4.5 decision | DFT pipeline scripts, workflow documentation, sample outputs. | DFT runs converge within tolerance on benchmark structures; uncertainties quantified. | T4.5 |
| T5.2 | MDIA & ALOA | Integrate DFT feedback loop with active learning orchestrator; define latency and queue management. | T5.1 pipeline, T4.3 orchestrator | Integration tests, latency report, queue policy doc. | Closed-loop dry run completes ≥3 iterations without manual intervention. | T5.1, T4.3 |
| T5.3 | QGMA | Update generative model training using real DFT feedback to refine property-conditional sampling. | T5.2 loop outputs | Updated model weights, performance report. | Demonstrated improvement in target property accuracy post-feedback. | T5.2 |
| T5.4 | BRA | Analyze DFT-informed loop performance vs simulator-only outcomes. | T5.2 logs, T4.4 metrics | Comparative analysis report, statistical tests. | Significant gains (p<0.05) in predictive accuracy or label-efficiency documented. | T5.2 |
| T5.5 | RKMA | Archive DFT workflows, datasets, and integration logs with reproducibility metadata. | T5.1-5.4 artefacts | Workflow package, provenance update. | Package regenerates end-to-end loop via scripted runbook. | T5.4 |

## M6 — Hardware Execution & Cost Management
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T6.1 | QHSOA | Benchmark candidate QPU and simulator backends (trapped-ion, superconducting) for circuit depth, noise, and cost. | T2.1 designs, vendor data | Backend comparison matrix, calibration schedule. | Includes decoherence, shot cost, queue times; PDA approves selected backends. | T2.1, T5.5 |
| T6.2 | QHSOA & QKAA | Transpile QSVR/QGPR circuits for selected hardware; apply error mitigation strategies. | T6.1 matrix, T2.3 models | Transpiled circuits, mitigation plan, validation metrics. | Hardware executions match simulator within tolerance; mitigation effectiveness quantified. | T6.1 |
| T6.3 | QHSOA & QGMA | Prepare generative circuits for hardware execution, optimizing for resource counts. | T6.1 matrix, T3.3 models | Hardware-ready generative circuits, resource usage report. | Circuits meet hardware limits; fidelity assessed via cross-entropy benchmarking. | T6.1 |
| T6.4 | QHSOA | Run pilot QPU experiments for regressors and generative models; capture execution logs and costs. | T6.2, T6.3 artefacts | Execution notebooks, cost dashboard, anomaly reports. | ≥3 successful hardware runs per model type; cost per iteration documented. | T6.2, T6.3 |
| T6.5 | PDA | Review hardware pilot results and update budget allocation. | T6.4 outputs | Hardware evaluation memo, updated budget tracker. | Memo includes go-forward recommendations; BRA acknowledges data integrity. | T6.4 |

## M7 — Benchmarking & Robustness Evaluation
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T7.1 | BRA | Define comprehensive metric suite (accuracy, label efficiency, novelty, robustness) and statistical testing plan. | T2.4, T4.4, T5.4 reports | Metric protocol document, hypothesis test plan. | PDA approves metric coverage; plan includes treatment of multiple comparisons. | T6.5 |
| T7.2 | BRA & DPQA | Execute robustness sweeps across noise models, feature-map perturbations, and DFT uncertainty. | T7.1 plan, T1.3 noise module | Robustness dataset, sensitivity plots, summary tables. | Identifies stability thresholds; significance annotated. | T7.1 |
| T7.3 | BRA & QHSOA | Compare hardware vs simulator performance including error mitigation impact. | T6.4 logs, T7.1 plan | Comparative report, variance decomposition analysis. | Hardware deviations quantified with confidence intervals; recommendations produced. | T6.4, T7.1 |
| T7.4 | BRA & ALOA | Benchmark QAL loop vs classical AL baselines using agreed metrics. | T4.4 loop, T2.4 baselines | Benchmark study, ablation results, CI tables. | Quantum approaches outperform baselines on ≥2 key metrics or provide clear trade-off analysis. | T7.1, T2.4 |
| T7.5 | PDA | Convene milestone review to confirm readiness for manuscript synthesis. | T7.2-7.4 outputs | Decision log, action register. | All critical gaps closed or mitigation plans assigned; go-ahead recorded. | T7.4 |

## M8 — Reproducibility, Packaging & Release Prep
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T8.1 | RKMA | Compile final provenance graph linking data, code, experiments, and hardware runs. | All prior milestones | Provenance report, interactive graph export. | Graph passes completeness audit; PDA signs reproducibility attestation. | T7.5 |
| T8.2 | RKMA & DPQA | Produce release-ready dataset/code bundles with containerized environments. | T8.1 graph, T1.6/T5.5 packages | Container images, environment manifests, checksum list. | End-to-end pipeline reproduces results on fresh environment; CI evidence archived. | T8.1 |
| T8.3 | RKMA & BRA | Draft transparency and ethics statement covering limitations, biases, and risk mitigations. | T7.2-7.4 analyses | Statement doc, risk mitigation appendix. | PDA approves for inclusion in manuscript; statement addresses compliance guidelines. | T7.5 |
| T8.4 | PDA | Final readiness review ensuring artefacts meet conference submission policies (data sharing, reproducibility). | T8.2 bundles, T8.3 statement | Readiness memo, submission checklist. | Memo confirms compliance with venue requirements; outstanding gaps resolved. | T8.2, T8.3 |

## M9 — Manuscript Authoring & Submission
| Task ID | Owner (Agent) | Description & Key Activities | Required Inputs | Deliverables | Acceptance Criteria | Dependencies |
| --- | --- | --- | --- | --- | --- | --- |
| T9.1 | RKMA (Lead Author) | Develop manuscript outline aligned with conference template; assign section owners. | T0.3 template, T7.5 decision | Detailed outline, author responsibility matrix, timeline. | PDA approves outline; all agents confirm responsibilities. | T8.4 |
| T9.2 | QKAA, QGMA, ALOA, MDIA, BRA | Draft technical sections (methods, experiments, results, discussion) incorporating figures and tables. | Outline, milestone artefacts | Draft sections, figure repository, table scripts. | Sections adhere to length/style guidelines; figures reproducible from scripts. | T9.1 |
| T9.3 | PDA & RKMA | Compose introduction, related work, conclusion, and impact statements highlighting contributions. | Draft sections, literature synth | Full narrative draft. | Narrative articulates novelty vs prior art; reviewers provide positive internal feedback. | T9.2 |
| T9.4 | All Agents | Conduct internal peer review (two-pass): technical validation, clarity edit, compliance check. | Draft manuscript, artefacts | Review reports, revision log. | All blocking comments resolved; compliance checklist satisfied. | T9.3 |
| T9.5 | RKMA | Finalize manuscript (LaTeX/Word), supplementary materials, and reproducibility appendix. | T9.4 revisions | Submission-ready package, supplementary files, README. | Cross-checked against venue checklist; zero outstanding TODOs. | T9.4 |
| T9.6 | PDA | Submit to selected conference; confirm receipt and archive submission artefacts. | Final package | Submission confirmation, archived ZIP, communication log. | Confirmation stored; follow-up plan for reviewer queries documented. | T9.5 |

## Governance & Tracking Notes
- PDA maintains rolling risk and decision logs; update after each milestone review.
- RKMA ensures all tasks conclude with provenance updates before closure.
- Use experiment tracking (e.g., MLflow, Weights & Biases) integrated via T4.3 to centralize metrics for manuscript figures.
- Employ fortnightly syncs where each agent reports status against acceptance criteria.

## Completion Definition
The project is considered complete when:
1. All milestone exit criteria are met and recorded in PDA decision logs.
2. Reproducibility attestations (T8.1-T8.4) and compliance checks are passed.
3. Manuscript submission (T9.6) is confirmed with all supplementary artefacts archived for post-review rebuttal or journal extension.

