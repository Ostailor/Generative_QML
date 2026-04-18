# Working Draft for Paper and Poster Development

**Working title:** Generative Quantum Machine Learning for Cost-Efficient, High-Strength Materials Discovery

**Short title:** Quantum active learning for cost-efficient alloy discovery

**Authors:** [Author names to be inserted]

**Affiliations:** [Affiliations to be inserted]

**Corresponding author:** [Name, email]

**Target venue:** [Conference or journal to be inserted]

**Version date:** 2026-04-18

**Editorial note:** This draft is designed to be easy to edit into either a conference paper, a preprint, or a research-poster narrative. It is grounded in repository artefacts current through the fresh real-DFT closure and the April 18, 2026 benchmark summary. Where the repository does not contain definitive metadata, placeholders are left explicitly rather than guessed.

## Claim guardrail

The strongest claims currently supported by the repository are:

- The workflow closes a full quantum-active-learning loop from candidate generation to real DFT feedback.
- The fresh production DFT campaign completed 12/12 jobs, returned 12 valid candidates, and achieved a label-efficiency gain of 0.90 against a 0.30 target.
- The broader quantum-ML stack shows positive benchmarking signals, but those results should be treated as supporting context rather than the main headline.

The repository does **not** currently justify strong claims about experimentally measured mechanical strength, synthesis success, or real-world manufacturability. The paper should therefore frame the contribution as a computational discovery and screening pipeline for candidate high-strength materials, not as final physical validation.

## Abstract

Computational materials discovery is often bottlenecked by the cost of first-principles labeling, especially when broad composition spaces must be screened before only a small subset can be advanced to detailed study. This project develops a modular quantum-active-learning workflow for high-entropy alloy discovery that combines quantum regression models, a property-conditioned generator, acquisition-driven candidate ranking, and real density-functional-theory feedback. The workflow is organized so that data curation, model training, candidate generation, active-learning selection, production DFT execution, benchmarking, and reproducibility tracking remain traceable and individually auditable.

The current repository evidence supports a fresh real-DFT closure for campaign `t5r4-20260211-fasttrack-221-mw4`. In that rerun, the production workflow executed three active-learning iterations, completed 12 of 12 planned DFT jobs, produced 12 valid candidates, and achieved a label-efficiency gain of 0.90 against an acceptance threshold of 0.30. Additional server-side non-QPU campaigns provide broader screening and benchmark context, including a 60-job screening campaign and preserved elastic-benchmark convergence outputs. Together, these results establish a reproducible and publication-ready baseline for cost-efficient candidate prioritization in alloy discovery while maintaining a clear distinction between real-backed headline claims and broader supporting analyses.

## Keywords

- quantum machine learning
- active learning
- high-entropy alloys
- density functional theory
- candidate generation
- reproducibility

## 1. Introduction

The central problem addressed by this project is the mismatch between the enormous size of modern materials search spaces and the cost of evaluating each candidate with first-principles methods. In high-entropy alloy discovery, even a chemically constrained search remains large enough that brute-force density-functional-theory screening becomes operationally expensive and slow. As a result, the practical research question is not simply how to predict material properties, but how to decide which candidate should receive the next expensive label.

This repository implements a modular answer to that problem. The proposed system combines three ideas. First, quantum regression models are used to learn structure-property relationships and estimate uncertainty. Second, a property-conditioned generator proposes new alloy compositions that remain within domain-informed chemistry constraints. Third, an acquisition stage ranks candidates so that expensive DFT runs are spent on the most informative or promising options rather than on uninformed enumeration.

The scientific motivation is twofold. At the algorithmic level, the work tests whether quantum-inspired or quantum-kernel models can participate meaningfully in an active-learning workflow without unacceptable accuracy penalties. At the systems level, it tests whether those models can be integrated into a fully traceable pipeline that includes real DFT execution, server-side benchmarking, and reproducibility packaging. The present evidence indicates that the second question now has a strong positive answer: the workflow runs end-to-end and produces auditable, real-backed campaign outputs.

## 2. Project Scope and What Has Been Completed

The repository documents a multi-milestone program running from data preparation through manuscript packaging. For the purpose of a paper, the most important completed work can be summarized as follows.

### 2.1 Data readiness and chemistry constraints

- Multi-source datasets covering high-entropy alloys and related materials families were curated and harmonized.
- Automated preprocessing and validation checks were implemented and released with manifests and quality reports.
- Domain constraints governing valid alloy suggestions were encoded so that candidate generation remains chemistry-aware.

### 2.2 Quantum regression and generation

- Quantum support vector regression and quantum Gaussian process regression components were built and benchmarked.
- A property-conditioned generator was implemented so candidate proposals can be filtered toward target property windows before expensive DFT runs.
- The generator and regressors were integrated into a common active-learning pipeline rather than evaluated only as isolated models.

### 2.3 Closed-loop active learning with production DFT

- A production-oriented DFT workflow was implemented with queue-aware execution, handoff packages, and campaign summaries.
- A fresh real-DFT campaign rerun was completed on February 11, 2026, producing new artefacts rather than relying only on historical logs.
- Release bundles, manifests, and reproduction checks were generated so the campaign can be audited and replayed.

### 2.4 Benchmarking, server-side campaign context, and paper packaging

- Additional server-side non-QPU screening and benchmark campaigns were recorded beyond the 12-job fresh rerun.
- Benchmark and robustness artefacts were refreshed and summarized on April 18, 2026.
- The repository already contains manuscript, poster, submission, and reproducibility packaging scaffolding that can be reused for publication.

## 3. Methods

### 3.1 Data curation and provenance

The pipeline begins with curated materials datasets and explicit provenance tracking. Preprocessing scripts harmonize source datasets and write release-ready artefacts with manifests, checksums, and validation summaries. The goal is not only data cleanliness but also auditability: each model result and campaign summary can be traced back to a dataset release and a documented preprocessing path.

This data layer serves two roles. First, it supplies model inputs for regression and generation. Second, it enforces stability at the pipeline level so that performance changes can be attributed to algorithmic or workflow changes rather than to silent data drift.

### 3.2 Domain constraints for alloy search

The repository encodes chemistry and composition guardrails so the generator does not operate over an unconstrained combinatorial space. These rules prevent obviously unrealistic compositions from propagating into downstream scoring or DFT handoff. The effect is practical as well as scientific: each filtered-out invalid candidate is one less opportunity to waste model capacity or simulation budget.

### 3.3 Quantum regression models

Two regression components are central to the workflow.

- A quantum support vector regressor is used to model structure-property relationships while exposing the workflow to a quantum-kernel representation.
- A quantum Gaussian process regressor adds uncertainty estimates so candidate ranking can account not just for predicted promise but also for where the model remains unsure.

The repository evidence shows that the quantum support vector regressor closely tracks the classical baseline on RMSE, while the quantum Gaussian process component provides calibrated uncertainty information with reported coverage near 0.90. Even where the quantum regressor does not outperform every classical comparator on every metric, it is operationally useful because it can participate in acquisition-driven selection without collapsing predictive performance.

### 3.4 Property-conditioned candidate generator

The generator produces new alloy compositions subject to target-property and chemistry-aware constraints. A key result already present in the repository is that the conditioned generator reaches a 100% compliance rate under the configured property-conditioning checks. That means the active-learning stage is not forced to clean up a large volume of implausible generator outputs after the fact.

This generator is important to the paper because it changes the framing from passive ranking to closed-loop discovery. The workflow is not merely choosing among a fixed candidate list; it is generating new candidates, then using model predictions and uncertainty estimates to decide which of those candidates deserve expensive follow-up.

### 3.5 Active-learning orchestration

The orchestration layer ranks candidates by combining predictive signals and uncertainty. In plain terms, the workflow asks two questions at once:

1. Which candidates already look promising?
2. Which candidates are likely to teach the model something important if simulated?

This stage is what turns regression and generation into a cost-management system rather than a disconnected modelling exercise. Repository artefacts from the pre-DFT loop report a 32% label-efficiency gain in the simulated setting, and the fresh real campaign reports a much larger 0.90 gain under the production DFT budget used for the February 2026 rerun.

### 3.6 Production DFT workflow

The DFT layer is the strongest real-backed component in the current repository state. The workflow builds per-candidate handoff packages, executes Quantum ESPRESSO calculations through a queue-managed production path, logs outcomes, and aggregates them at campaign level. Importantly, the fresh rerun was not treated as an ornamental reproduction step. Repository notes describe concrete engineering fixes made during the rerun, including queue isolation and stricter validation semantics, which materially improved the auditability and correctness of the production workflow.

### 3.7 Server-side benchmark and convergence context

Beyond the main discovery loop, the repository includes larger non-QPU server campaigns and preserved QE benchmark artefacts that expose runtime, screening progression, and convergence behavior more directly than the original summary plots did. These artefacts should be presented carefully. They strengthen the systems story and show how the workflow behaves at larger scale, but they should not overshadow the real-DFT campaign in the paper’s primary narrative.

## 4. Experimental Setup

### 4.1 Candidate generation and campaign scale

The fresh real-DFT campaign summary reports:

- 3 planned and executed active-learning iterations
- 150 conditioned candidates upstream of the real campaign
- 12 candidates selected for real DFT
- 12 completed jobs
- 0 failed jobs
- 12 valid candidates

This means the real campaign spent expensive labels on only a small subset of the broader generated pool, which is consistent with the project’s stated objective of reducing screening cost.

### 4.2 Label budgets

For the fresh real campaign:

- Classical reference budget: 120 labels
- Quantum-guided budget actually used: 12 labels
- Reported label-efficiency gain: 0.90

For the earlier simulated active-learning setting:

- Classical label budget: 1000
- Quantum label budget: 680
- Reported label-efficiency gain: 0.32

The paper should distinguish these two contexts clearly. The 0.90 value is the real-backed campaign result and should lead the headline. The 0.32 value belongs in the supporting narrative about the earlier active-learning loop.

### 4.3 Regression and generator metrics

Selected repository metrics that are useful for the methods and results sections are:

- QSVR RMSE: 1.4242962889041533
- Classical RMSE comparator for that task: 1.4242723611887744
- QGPR reported coverage: 0.90
- Generator property-compliance rate: 1.0
- Generator novelty gap versus classical baseline: 0.2264000752915899
- Generator feasibility gap versus classical baseline: 0.11666666666666659

### 4.4 Server-side screening and convergence context

The server-side non-QPU campaign stack now provides a broader context than the original 12-job rerun alone. In particular:

- `top-tier-nonqpu-gpu-20260306T211650Z-scf-20260306t211651z` completed 60 of 60 screening jobs across 15 iterations.
- `top-tier-nonqpu-gpu-20260303T195828Z-vc-20260303t195829z` completed 23 of 24 variable-cell relaxations.
- `BENCH-ELASTIC-0001` preserves per-state QE convergence metadata suitable for a genuine convergence figure.

These artefacts are more appropriate for the paper than the earlier simulated hardware-adapter summary because they come directly from the server-side DFT workflow rather than from a synthetic backend-adapter pilot.

## 5. Results

### 5.1 Real-backed headline results

The strongest paper-ready result is the fresh real-DFT campaign closure. Campaign `t5r4-20260211-fasttrack-221-mw4` executed three active-learning iterations and produced complete closure under the configured screening profile. Specifically:

- 12 of 12 planned DFT jobs completed
- 0 jobs failed
- 12 valid candidates were returned
- label-efficiency gain reached 0.90

This exceeds the campaign acceptance threshold of 0.30 and provides a concrete, real-backed answer to the project’s main systems question: the full discovery loop can operate end-to-end while sharply reducing the number of expensive labels consumed.

### 5.2 Benchmark and robustness context

The broader benchmark context now has two useful layers:

1. Repository-level summary metrics:

- real label-efficiency gain: 0.9
- novelty gap: 0.2264000752915899
- sensitivity index: 0.039025751745155185
- quantum-versus-classical gap: 0.6000000000000001

2. Server-side campaign context:

- 60 completed screening jobs from the March 6, 2026 non-QPU campaign.
- Best-so-far screening progress visible over 15 active-learning iterations.
- QE convergence metadata preserved across the elastic benchmark states.

These results suggest that the workflow remains competitive and operationally stable under broader benchmark views, but the paper should preserve the existing headline policy encoded in the repository: real-backed campaign claims belong in the main narrative; broader supporting artefacts belong in secondary roles unless directly tied to the real campaign.

### 5.3 Candidate alloys for emphasis

The repository already contains candidate compositions that can be highlighted in the paper or used later in a better poster.

#### Real-campaign candidates worth mentioning

| Candidate ID | Composition | Why it matters |
| --- | --- | --- |
| FAST-009 | Al0.25 Co0.25 Fe0.20 Ni0.30 | Strongest formation-energy result among the fresh real-campaign candidates. |
| FAST-010 | Al0.22 Co0.28 Fe0.25 Ni0.25 | Strong stability-oriented result with competitive density. |
| FAST-011 | Al0.28 Co0.22 Fe0.25 Ni0.25 | Combines favorable formation energy with moderate density. |
| FAST-007 | Al0.30 Co0.25 Fe0.20 Ni0.25 | One of the lowest-density validated real-campaign candidates. |
| FAST-003 | Al0.30 Co0.20 Fe0.25 Ni0.25 | Balanced candidate combining low density and strong real-campaign performance. |

#### Generator follow-up candidates worth mentioning

| Candidate ID | Composition | Why it matters |
| --- | --- | --- |
| QGAN-071 | Nb0.39 Al0.39 Nd0.21 | Best density-match proposal in the conditioned generator pool. |
| QGAN-029 | Mg0.37 W0.19 Mn0.08 Cr0.19 Y0.08 Co0.08 | A chemistry-diverse BCC proposal with extremely small density error. |
| QGAN-149 | Y0.06 Hf0.06 Si0.06 Zn0.10 Sn0.19 Ti0.06 Mo0.06 Ga0.34 V0.06 | Strong near-target conditioned density with broader compositional diversity. |

These tables should be treated as computational candidate highlights, not as experimentally validated materials claims.

## 6. What Is Novel in This Work

The paper can reasonably claim novelty in the **integration** of components, even where individual techniques have precedents.

### 6.1 Novelty in workflow integration

The distinctive contribution is not just a quantum model, not just a generator, and not just a DFT pipeline. It is the combination of:

- quantum regression models for predictive and uncertainty-aware ranking,
- property-conditioned candidate generation,
- acquisition-driven closed-loop selection,
- and real production DFT feedback under a reproducible campaign structure.

### 6.2 Novelty in claim discipline

Another strength of the repository is methodological restraint. The project distinguishes between:

- real-backed production campaign evidence,
- benchmark and robustness context,
- and historical simulator-only ablation material.

That distinction makes the work more defensible than a paper that collapses all evidence types into a single performance headline.

### 6.3 Novelty in operational maturity

The repository goes beyond algorithm demonstration by including manifests, release bundles, validation logic, campaign summaries, and reproduction checks. This matters because it turns the project into a credible research system rather than a collection of disconnected experiments.

## 7. Discussion

The current evidence suggests that the main achievement of the project is cost-efficient prioritization, not ultimate physical validation of final material performance. In practical terms, the pipeline appears to be successful at choosing which candidates deserve expensive first-principles evaluation and at packaging the resulting evidence so it can be audited, benchmarked, and reused.

The real-DFT campaign result is especially important because it validates the operational side of the project. Many ML-for-materials papers stop at model benchmarking or offline screening. Here, the workflow actually crosses the boundary into queue-managed production DFT execution and returns a campaign-level summary that can support a strong systems contribution.

The broader benchmark signals are encouraging but should be handled carefully. The novelty advantage, the larger server-side screening campaign, and the preserved QE convergence benchmarks all strengthen the case that the research direction is worth pursuing. However, the strongest argument remains the real campaign itself.

## 8. Limitations and Claim Boundaries

The paper should state the following limitations explicitly.

1. The fresh production campaign used a screening-oriented DFT profile to keep runtime manageable. This supports throughput claims better than ultimate high-precision property claims.
2. The repository currently supports computational screening and prioritization claims more strongly than end-to-end claims about experimentally measured high strength.
3. Larger server-side non-QPU campaigns provide operational and benchmarking context, but they do not replace physical materials validation.
4. Some candidate compositions highlighted in this draft come from generator outputs and should be presented as follow-up opportunities, not completed validation results.

These limitations do not weaken the main paper if framed correctly. They simply define the paper’s honest scope.

## 9. Conclusion

This project now supports a credible paper centered on a modular quantum-active-learning workflow for alloy discovery. The repository contains evidence for complete data-to-DFT closure, a fresh real-DFT campaign rerun, broader server-side screening and convergence context, and reproducibility packaging. The strongest defensible headline is that the workflow reduced expensive labeling demand while preserving traceability and delivering valid real-DFT candidate outputs. That is already a publishable systems-level contribution, even before any stronger experimental materials-validation claims are added in future work.

## 10. Suggested Figures and Tables for the Paper

### Suggested main figures

1. **Pipeline overview figure**
   - Data curation -> regression -> generation -> acquisition -> real DFT -> retraining
   - Can be adapted from the repository architecture and walkthrough docs

2. **Server-side candidate landscape**
   - Use `docs/poster/figures/fig_server_side_candidate_landscape.png`
   - Shows the density/formation-energy trade-off across the 60-job non-QPU screening campaign
   - Suitable as a main-results or supporting-results figure

3. **Server-side screening progress**
   - Use `docs/poster/figures/fig_server_side_screening_progress.png`
   - Shows best-so-far screening progress over completed jobs
   - Best placed in the results or discussion section

4. **QE convergence benchmark figure**
   - Use `docs/poster/figures/fig_qe_elastic_benchmark_convergence.png`
   - Shows SCF-step and final-error behavior across elastic benchmark states
   - Best placed in discussion, methods validation, or appendix

### Suggested tables

1. **Dataset and constraint summary**
2. **Model component summary**
3. **Real campaign outcomes**
4. **Highlighted candidate compositions**
5. **Claim boundary table**
   - main-text headline claims
   - supporting benchmark claims
   - explicitly out-of-scope claims

## 11. Suggested Section-to-Poster Mapping

If you later turn this into a better poster, the cleanest extraction is:

- Title + one-sentence motivation from the Introduction
- Five-step plain-English loop from Methods
- Real-campaign KPI block from Results
- Four highlighted candidate alloys from the candidate table
- One short limitations box from Section 8

That keeps the poster tied to the strongest evidence instead of trying to compress the whole paper.

## 12. Remaining Edits Before Submission

- Insert authors, affiliations, acknowledgements, and funding details.
- Convert placeholder local-source references into venue-formatted citations.
- Decide whether the paper should target a systems-oriented venue, a quantum-computing venue, or a materials-informatics venue.
- Confirm whether any additional mechanical-property or experimental-validation data exists outside this repository and should be incorporated.
- Replace or supplement raster figures with native editable charts where useful.
- Add a formal reference list from `docs/literature/annotated_bibliography.md`, `docs/literature/synthesis_memo.md`, and your BibTeX/Zotero source.

## 13. References Placeholder

The following source groups should be converted into venue-style references in the final manuscript:

- quantum kernel learning for regression in materials and scientific ML
- Gaussian-process-based active learning and Bayesian optimization
- generative modeling for materials discovery
- high-entropy alloy design and screening literature
- DFT workflow and reproducibility best practices
- DFT convergence and workflow benchmarking references relevant to the benchmark section

Repository starting points:

- `docs/literature/annotated_bibliography.md`
- `docs/literature/synthesis_memo.md`
- `docs/venue_target_brief.md`
- `docs/templates/IEEE_QCE/README.md`
- `docs/templates/npj_Quantum_Materials/README.md`

## Appendix A. Completed Milestones Summary

If you need an operational appendix or a project-status section for a thesis-style document, the following completion summary is already supported by the repository:

- M0: program alignment and literature grounding completed
- M1: data readiness and chemistry constraints completed
- M2: quantum regression foundations completed
- M3: quantum generative modeling capability completed
- M4: active-learning loop design completed
- M5-real: fresh production DFT rerun completed with new artefacts
- M6: workflow benchmarking artefacts present
- M7: benchmarking and robustness artefacts present
- M8: reproducibility packaging present
- M9: manuscript, poster, submission, and preprint packaging present

This appendix is optional for a paper but may be useful for internal reporting or a dissertation chapter.
