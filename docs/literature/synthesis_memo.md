# Literature Synthesis Memo — T0.2

## Aim 1: Modular QAL Pipeline with QSVR/QGPR
1. **Prioritize expressive yet trainable feature maps** — Combine hardware-efficient circuits validated by Havlíček *et al.* (2019) with inductive bias diagnostics from Kübler *et al.* (2021) to avoid barren kernels.
2. **Adopt kernel-based uncertainty estimates** — Integrate hybrid QGPR schemes (Liu *et al.*, 2021) to deliver calibrated variance for acquisition scoring.
3. **Leverage quantum-classical hybridization** — Use classical shadow-based kernel estimation (Jerbi *et al.*, 2023) to reduce QPU shot requirements while retaining quantum expressivity.
4. **Benchmark against theoretical advantage frontiers** — Map experiments against data-dependent advantage criteria highlighted by Huang *et al.* (2021) and Liu *et al.* (2021) to justify QPU usage.
5. **Document regression-focused implementations early** — Extend QSVR frameworks from Xu *et al.* (2019) and Zhao *et al.* (2015) into reusable modules, enabling fast iteration during M2.

## Aim 2: Benchmarking on Materials Datasets & High-Entropy Alloys
1. **Curate HEA descriptors aligned with actives** — Use property descriptors from Jian *et al.* (2021) and Zhang *et al.* (2014) to seed DPQA pipelines with domain-informed features.
2. **Adopt AL baselines from established studies** — Reproduce workflows from Lookman *et al.* (2019) and Xue *et al.* (2016) to provide classical comparison curves for label efficiency.
3. **Exploit HEA-focused constraint heuristics** — Encode phase stability heuristics from Natarajan *et al.* (2020) and Hinojosa-Romero *et al.* (2022) into MDIA rule engines.
4. **Integrate Bayesian optimization best practices** — Apply acquisition tuning strategies from Cao *et al.* (2022) to maintain convergence in high-dimensional composition space.
5. **Plan dataset augmentation via public repositories** — Align with Materials Project (Jain *et al.*, 2013) and OQMD (Tran *et al.*, 2016) formats for reproducible data sharing.

## Aim 3: Robustness, Noise, and Acquisition Strategies
1. **Embed error mitigation hooks** — Implement zero-noise extrapolation (Temme *et al.*, 2017) and measurement mitigation (Nation *et al.*, 2021) in QAL runs scheduled on hardware.
2. **Quantify noise impact on generalization** — Compare kernel robustness predicted by Banchi *et al.* (2021) and Thanasilp *et al.* (2023) with empirical HEA performance metrics.
3. **Design acquisition functions resilient to noise** — Utilize Bayesian optimization reviews (Shahriari *et al.*, 2016) to select acquisition policies tolerant to stochastic labels from noisy QGPR outputs.
4. **Cross-check hardware vs simulator metrics** — Use methodologies from Kandala *et al.* (2019) to benchmark fidelity gaps and calibrate cost models in M6.
5. **Maintain adaptive scheduling** — Borrow sequential design heuristics from Snoek *et al.* (2012) to dynamically adjust exploration/exploitation balance based on uncertainty profiles.

## Aim 4: Quantum & Classical Generative Models for Materials
1. **Prototype variational quantum generators** — Start with VQC-based models (Romero *et al.*, 2019; Benedetti *et al.*, 2019) before integrating adversarial training loops.
2. **Incorporate QGAN training stabilities** — Apply training protocols from Zoufal *et al.* (2019) and Lloyd & Weedbrook (2018) to manage gradient pathologies on QPUs.
3. **Condition on materials descriptors** — Extend generative conditioning techniques from Noh *et al.* (2019) and Court *et al.* (2020) to embed HEA property targets.
4. **Bridge classical and quantum models** — Use Acta Materialia review by Liu *et al.* (2018) to benchmark quantum generators against state-of-the-art classical alternatives.
5. **Define novelty metrics** — Adopt evaluation schemes from Wang *et al.* (2020) and G-SchNet results to quantify diversity vs feasibility in generated compositions.

## Aim 5: DFT Integration, Automation, and Reproducibility
1. **Standardize DFT toolchain** — Anchor workflows on Quantum ESPRESSO (Giannozzi *et al.*, 2009) and VASP (Kresse & Furthmüller, 1996) for compatibility with community standards.
2. **Automate input/output via pymatgen** — Employ Ong *et al.* (2013) utilities to harmonize structure representations and property extraction.
3. **Adopt high-throughput orchestration patterns** — Use AFLOW (Curtarolo *et al.*, 2013) as template for queue management and metadata capture.
4. **Integrate Bayesian optimizers with DFT** — Implement Phoenics (Häse *et al.*, 2018) concepts to manage expensive feedback loops under uncertainty.
5. **Ensure provenance via public repositories** — Mirror data schemas from Materials Project and OQMD to facilitate open artefact releases and reproducibility audits.

## Summary
The literature base substantiates a modular QAL pipeline that leverages quantum kernels for supervised regression, integrates HEA-specific active learning strategies, and incorporates robust generative modeling with automated DFT feedback. The actionable insights above translate directly into upcoming tasks in `TASKS.md`, providing evidence-backed guardrails for experimental design, robustness evaluation, and reproducibility planning.

