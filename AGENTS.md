# AGENTS.md

## Document Purpose
This file defines the autonomous and human-in-the-loop agent stack that will execute the "Generative Quantum Machine Learning for Cost-Efficient, High-Strength Materials Discovery" program. Each agent description specifies remit, decision scope, interfaces, and expected artefacts so that future planning in `TASKS.md` can assign workstreams with research-grade granularity, quality gates, and reproducibility guardrails.

## Research Program Snapshot
- **Core objectives**: Build a modular quantum-active-learning (QAL) pipeline combining QSVR/QGPR, property-conditional quantum generative models, and DFT feedback to discover high-entropy alloys (HEAs) and other advanced materials with reduced simulation cost.
- **Key evaluation axes**: predictive accuracy, label-efficiency, robustness to noise/feature-map perturbations, generative novelty, and real-QPU vs simulator trade-offs.
- **Critical differentiators**: Integration of quantum kernels into active learning, closed-loop coupling to DFT workflows, and benchmarking against classical AL baselines across diverse datasets (perovskites, HEAs, doped nanoparticles).

## Agent Stack Overview
| Agent ID | Focus Area | Primary Deliverables |
| --- | --- | --- |
| PDA | Program Director & Alignment | Research roadmap, milestone gates, cross-agent arbitration |
| QKAA | Quantum Kernel & Regressor Architect | QSVR/QGPR designs, kernel studies, feature-map specifications |
| QGMA | Quantum Generative Modeling | Property-conditional generative circuits, sampling protocols |
| ALOA | Active Learning Orchestrator | Acquisition strategies, closed-loop scheduling, exploration/exploitation tuning |
| MDIA | Materials & DFT Integration | HEA design rules, DFT workflows, physics-informed constraints |
| DPQA | Data Pipeline & Quality | Dataset curation, preprocessing scripts, uncertainty & noise models |
| QHSOA | QPU & Simulation Operations | Backend selection, calibration logs, cost tracking |
| BRA | Benchmarking, Robustness & Baselines | Metric suites, classical baselines, sensitivity analyses |
| RKMA | Reproducibility & Knowledge Management | Lab notebooks, provenance graphs, documentation standards |

## Agent Charters & Interfaces

### PDA — Program Director Agent
- **Mandate**: Own the scientific vision, confirm that agent outputs satisfy aims (modular QAL pipeline, benchmarking, robustness, generative integration).
- **Key decisions**: Approve milestone criteria, prioritize datasets, allocate QPU credit budget, adjudicate trade-offs between accuracy vs cost.
- **Inputs**: Progress briefs from every agent, risk registers, updated literature scans.
- **Outputs**: Quarterly research roadmap, decision logs, escalation memos for unresolved uncertainties.
- **Task cues for `TASKS.md`**: Use PDA when scoping new experiments, redefining milestones, or resolving conflicts between algorithmic and physical constraints.

### QKAA — Quantum Kernel & Regressor Architect Agent
- **Mandate**: Design, implement, and benchmark QSVR/QGPR models with parameterized quantum kernels across simulators and QPUs.
- **Responsibilities**:
  - Derive feature-map parameterizations tailored to materials descriptors (composition vectors, symmetry features, local environments).
  - Quantify kernel expressivity, trainability, and noise resilience; deliver comparative studies against classical kernels.
  - Collaborate with ALOA on model integration within the active-learning loop and with QHSOA for backend-aware circuit transpilation.
- **Artefacts**: Kernel design briefs, circuit definitions, regression performance reports with uncertainty quantification.
- **Quality gates**: Provide reproducible notebooks, cite hardware calibration data, and report gradient-based sensitivity for feature-map perturbations.

### QGMA — Quantum Generative Modeling Agent
- **Mandate**: Construct property-conditional quantum generative models that synthesize candidate HEA compositions satisfying target properties.
- **Responsibilities**:
  - Evaluate variational generative circuits (e.g., data re-uploading, quantum Boltzmann machines) for mixed discrete-continuous chemical spaces.
  - Imbue generated structures with chemical validity constraints from MDIA and integrate feedback signals from DFT evaluations.
  - Produce sampling strategies that balance novelty with adherence to predicted property thresholds.
- **Artefacts**: Generative model specs, sampling diagnostics, diversity vs feasibility metrics, candidate libraries for downstream screening.
- **Quality gates**: Demonstrate mode coverage metrics, maintain rejection reasons for invalid proposals, and log entanglement/resource requirements for QPU execution.

### ALOA — Active Learning Orchestrator Agent
- **Mandate**: Architect the closed-loop QAL workflow that fuses quantum regressors, generative proposals, and acquisition policies.
- **Responsibilities**:
  - Select and tune Bayesian optimization and information-theoretic acquisition functions suited to high-dimensional HEA search spaces.
  - Schedule iterative loops: query generation → DFT evaluation → model update, ensuring synchronization across QKAA, QGMA, and MDIA deliverables.
  - Quantify label-efficiency gains (30–50% reduction target) and evaluate exploration vs exploitation balance.
- **Artefacts**: Loop schematics, acquisition policy rationales, convergence diagnostics, ablation studies on active learning settings.
- **Quality gates**: Provide statistical significance analyses, maintain experiment registries, and track compute budgets consumed per iteration.

### MDIA — Materials & DFT Integration Agent
- **Mandate**: Inject domain expertise on HEAs and related materials, orchestrate DFT workflows, and enforce physical plausibility constraints.
- **Responsibilities**:
  - Curate thermodynamic and mechanical property targets; define composition priors and phase stability heuristics.
  - Implement automated DFT pipelines (structure relaxation, property extraction) and return uncertainty-aware feedback to ALOA.
  - Validate generative outputs for chemical feasibility and interface with BRA on evaluation metrics grounded in materials science.
- **Artefacts**: DFT workflow blueprints, validation reports, constraint libraries, data provenance for experimental/DFT sources.
- **Quality gates**: Ensure DFT calculations include convergence criteria, document approximations (e.g., exchange-correlation functionals), and tag property uncertainties.

### DPQA — Data Pipeline & Quality Agent
- **Mandate**: Guarantee high-quality, versioned datasets feeding all quantum and classical models.
- **Responsibilities**:
  - Aggregate datasets (perovskites, HEAs, doped nanoparticles) with metadata on source credibility and preprocessing steps.
  - Engineer feature representations (graph-based descriptors, composition embeddings) and simulate perturbations/noise scenarios for robustness studies.
  - Maintain data lineage, schema contracts, and access layers for reproducible experiments.
- **Artefacts**: Data dictionaries, preprocessing scripts, noise-injection protocols, validation dashboards.
- **Quality gates**: Automated checks for data drift, missingness, and bias; hashed dataset releases with semantic versioning.

### QHSOA — QPU & Simulation Operations Agent
- **Mandate**: Operate hybrid execution across quantum simulators and real QPUs, optimizing cost, fidelity, and scheduling.
- **Responsibilities**:
  - Benchmark circuit performance across trapped-ion and superconducting backends, tracking decoherence and runtime costs.
  - Provide transpilation strategies, error mitigation plans, and hybrid execution policies (shot allocation, batching).
  - Coordinate with PDA on budget usage and with QKAA/QGMA on hardware-aware circuit revisions.
- **Artefacts**: Backend comparison matrices, calibration notebooks, execution logs, mitigation performance summaries.
- **Quality gates**: Record hardware configuration snapshots, include cost-per-experiment analyses, and flag hardware-induced anomalies to BRA.

### BRA — Benchmarking, Robustness & Baselines Agent
- **Mandate**: Deliver rigorous evaluation spanning quantum vs classical approaches, robustness assessments, and statistical reporting.
- **Responsibilities**:
  - Define metrics (RMSE, MAE, calibration error, acquisition efficiency, novelty score) and acceptable confidence intervals.
  - Implement classical active learning baselines (Gaussian processes, random forest AL, deep kernel learning) for comparative studies.
  - Run robustness tests for noise, feature-map variations, and DFT feedback perturbations; report uncertainties and sensitivity indices.
- **Artefacts**: Benchmark reports, statistical significance analyses, robustness heatmaps, comparison dashboards.
- **Quality gates**: Use reproducible evaluation scripts, document hypothesis tests, and maintain traceable seeds/configurations.

### RKMA — Reproducibility & Knowledge Management Agent
- **Mandate**: Preserve research-grade documentation, knowledge transfer, and compliance with open-science best practices.
- **Responsibilities**:
  - Maintain living lab notebooks, changelogs, and decision registries linking experiments to code/data revisions.
  - Enforce documentation templates for every agent, including metadata, dependencies, and verification status.
  - Curate literature reviews, standards, and reference implementations to inform future iterations of the project.
- **Artefacts**: Structured notebooks, documentation templates, knowledge graphs, publication-ready summaries.
- **Quality gates**: Validate that every experiment has replication instructions, ensure FAIR data principles, and audit provenance trails quarterly.

## Cross-Agent Workflows
1. **Data & Domain Intake**: DPQA prepares datasets and quality reports → MDIA annotates physical constraints → PDA validates scope.
2. **Model Design Loop**: QKAA and QGMA develop models with QHSOA hardware input → ALOA integrates into active-learning loop.
3. **Closed-Loop Execution**: ALOA schedules iterations → MDIA performs DFT → QKAA/QGMA update models → BRA evaluates outcomes → PDA reviews milestone readiness.
4. **Benchmarking & Reporting**: BRA runs comparative studies → RKMA codifies results → PDA and ALOA adjust strategies based on evidence.
5. **Knowledge Dissemination**: RKMA publishes internal briefs → informs DPQA/MDIA on data updates and ensures reproducibility packages are complete.

## Task Planning Guidance for `TASKS.md`
- **Granularity**: Map each task to a single primary agent; include co-agent consults where dependencies exist (e.g., ALOA task requiring DPQA inputs).
- **Inputs/Outputs**: Explicitly specify prerequisite artefacts and expected deliverables per task to preserve closed-loop continuity.
- **Quality Checks**: Reference the relevant agent quality gates so tasks embed acceptance criteria (e.g., BRA tasks must include statistical significance thresholds).
- **Sequencing**: Align tasks with cross-agent workflow order to avoid deadlocks (e.g., do not schedule QPU runs before QHSOA approves calibration).
- **Documentation Hooks**: Every task must include a handoff to RKMA for logging results and updating provenance records.

## Research-Grade Standards Checklist
- Use peer-reviewed references and state-of-the-art benchmarks in agent outputs.
- Enforce uncertainty quantification and error analysis across models and DFT evaluations.
- Preserve end-to-end traceability from raw data through quantum experiments to reported metrics.
- Plan for hardware variability by maintaining sandboxed simulator baselines before deploying to QPUs.
- Treat negative results and failure modes as first-class artefacts requiring documentation and root-cause analysis.

