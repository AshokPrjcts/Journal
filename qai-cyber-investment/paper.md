# Quantum-Inspired Artificial Intelligence for Cybersecurity Investment Prioritization: A Hybrid Decision Intelligence Framework

**Article type:** Short communication

---

**Author:** [Author Name]<sup>a,*</sup>

<sup>a</sup> [Department, Institution, Full Postal Address, Country]

<sup>*</sup> Corresponding author. E-mail address: [author@institution.edu]

---

## Highlights

- A hybrid decision-intelligence framework prioritizes cybersecurity investments.
- Learning, knowledge rules and a quantum-inspired optimizer are unified.
- Defence-in-depth synergies make budget allocation a quadratic problem.
- QIEA reaches a 0.06% gap to the best solution, beating greedy and GA.
- Ablation shows each layer adds value, lifting risk reduction by 7.5%.

---

## Abstract

Chief information security officers must allocate finite budgets across competing security controls, yet prevailing practice relies on ranked heuristics that ignore interactions between controls. We present a hybrid *decision-intelligence* framework for cybersecurity investment prioritization that couples three cooperating layers: (i) a data-driven prediction layer that learns per-control breach-mitigation effectiveness; (ii) a knowledge layer that encodes expert security rules through fuzzy amplification to recover value the predictor cannot observe from marginal features; and (iii) a quantum-inspired evolutionary optimizer (QIEA) that solves the budget-constrained selection problem. Because co-funding complementary controls yields super-additive "defence-in-depth" protection, the allocation problem is quadratic and non-submodular, which defeats myopic value-density heuristics. On a reproducible 120-control portfolio averaged over twenty independent seeds, the full framework attains a mean optimality gap of 0.06% to the best solution found, compared with 0.54% for a classical genetic algorithm, 1.84% for value-density greedy selection, and 2.58% for an exact linear optimizer that is blind to synergies. A layer-wise ablation confirms that each component contributes positively, improving realized annual risk reduction by 7.5% over an optimization-only baseline and raising return on security investment to 144.2%. The framework is implemented in dependency-free Python and released openly, offering security managers a transparent, auditable decision-support tool. Results indicate that modelling control synergies, rather than optimizer choice alone, is the decisive factor for effective cybersecurity budgeting.

**Keywords:** cybersecurity; investment; quantum-inspired; optimization; knowledge; decision-support; prioritization

---

## 1. Introduction

Organizations face an expanding threat surface under hard budget constraints, forcing security leaders to decide *which* controls to fund rather than *whether* to invest [1,2]. In practice these decisions are dominated by qualitative risk matrices and single-criterion rankings that score each control independently [3]. This independence assumption is unsafe: security controls interact. Endpoint detection is far more valuable alongside centralized log analytics; identity hardening compounds with privileged-access management. Such "defence-in-depth" complementarities make the realized value of a control portfolio *super-additive*, so any method that ranks controls in isolation systematically misallocates capital [3,4].

A credible decision-support system for this problem must therefore satisfy three requirements simultaneously: it must (i) *estimate* how effective each control is from data, (ii) *incorporate expert knowledge* that the data alone cannot express, and (iii) *optimize* the joint, interaction-aware objective under a budget. Each requirement maps onto a mature strand of artificial intelligence — machine learning, knowledge-based systems, and computational-intelligence optimization — yet the literature addresses them in isolation. Security-economics models optimize additive utilities but do not learn effectiveness from data [1,2,3]; knowledge-based risk tools reason over expert rules but do not optimize a budgeted portfolio [4]; and quantum-inspired optimizers solve generic knapsack problems without any domain learning or knowledge layer [5,6,7]. **No existing method unifies learning, knowledge, and interaction-aware optimization for cybersecurity investment, and none treats the allocation as the non-additive problem it actually is.** This is the gap we address.

We contribute a *hybrid decision-intelligence framework* that closes this gap. Its optimization core is a **quantum-inspired evolutionary algorithm (QIEA)**: a classical algorithm that represents each funding decision as a qubit in superposition and uses a rotation-gate update to emulate quantum amplitude amplification, giving strong global search on commodity hardware without quantum computers [5,6]. The framework's novelty is threefold — a new *problem formulation*, a new *architecture*, and a counter-intuitive *empirical finding*. Specifically, our contributions are:

1. **Formulation.** We cast cybersecurity budgeting as a **quadratic, synergy-aware** knapsack problem (Eq. 2) that explicitly models defence-in-depth complementarities, departing from the additive formulations that dominate prior work (Section 3.1).
2. **Architecture.** We design a three-layer pipeline — data-driven prediction, knowledge-based fuzzy correction, and quantum-inspired optimization — that converts raw threat data into transparent, auditable investment recommendations (Sections 3.2–3.4), to our knowledge the first application of a QIEA to this task.
3. **Evidence.** Through a reproducible 20-seed study we show the framework attains a 0.06% mean gap to the best solution found, and we establish that *modelling synergies, not the choice of optimizer, is the decisive factor* — an exact optimizer of the additive objective is beaten by heuristics that reason over interactions (Section 4).

The remainder of the paper is organized as follows. Section 2 reviews related work; Section 3 formalizes the problem and the three-layer framework; Section 4 reports the experimental evaluation; and Section 5 discusses implications, limitations, and conclusions.

## 2. Related work

**Security economics and investment.** Gordon and Loeb established the economic foundations of security investment [1], followed by portfolio and game-theoretic treatments [2,3]. Knowledge-based and multi-criteria methods have since been applied to security evaluation, including hesitant-fuzzy AHP–TOPSIS models for security–durability assessment [8], while machine-learning-based intrusion detection has been studied extensively for IoT networks [9]. These works largely assume additive utilities, rank alternatives independently, or target detection rather than budgeting, and do not provide an operational tool for budgeted portfolio selection.

**Machine learning for security risk.** Data-driven models are now central to threat detection and risk estimation: recent surveys and empirical studies cover machine-learning-based IoT security [10], ensemble intrusion detection [11], machine-learning-based anomaly detection in server logs [12], and unsupervised anomaly detection over streaming cybersecurity logs [13]. We adopt a comparable data-driven predictor to estimate per-control mitigation effectiveness, but embed it within an optimization pipeline rather than using it for detection alone.

**Knowledge-based and explainable decision support.** Expert systems and fuzzy inference have long supported risk assessment under uncertainty [4,14], and remain central to the scope of knowledge-based systems; recent instances include fuzzy unified decision techniques for healthcare security risk [15] and fuzzy-logic assessment for the industrial IoT [16]. Explainability is increasingly emphasized in AI-based decision support [17], and transparent, auditable recommendations are a core requirement for security governance. We adopt fuzzy amplification as the mechanism by which domain rules correct data-driven estimates, and our layered decomposition makes each recommendation explainable. The broader motivation—securing the digital transformation of enterprises—is well documented [18].

**Quantum-inspired and combinatorial optimization.** Han and Kim introduced quantum-inspired evolutionary algorithms for combinatorial problems [5], with subsequent surveys and advances in knapsack, feature-selection, and multi-objective tasks [6,7,19,20]. Portfolio-style selection has a rich metaheuristic literature [21], and the broader quantum-optimization toolkit—QUBO formulations [22] and quantum-inspired search—offers principled ways to handle non-additive objectives. Deep and federated anomaly-detection models for zero-day cyberattacks further motivate data-driven security analytics [23]. We are, to our knowledge, the first to apply a QIEA to synergy-aware cybersecurity investment prioritization, and to combine it with a learning and knowledge layer in a single framework.

## 3. Problem formulation and framework

### 3.1 Decision problem

Let *C* = {*c*<sub>1</sub>, …, *c*<sub>n</sub>} be candidate controls. Control *c*<sub>i</sub> has cost *k*<sub>i</sub>, protected asset value *a*<sub>i</sub>, annual breach probability *p*<sub>i</sub>, and an unknown mitigation effectiveness *m*<sub>i</sub> ∈ [0,1]. A decision vector *x* ∈ {0,1}<sup>n</sup> selects controls subject to budget *B*. The expected annual loss reduction (ELR) of a single control is

> ELR(*c*<sub>i</sub>) = *a*<sub>i</sub> · *p*<sub>i</sub> · *m*<sub>i</sub>.  (1)

Defence-in-depth means that funding a complementary pair (*i*, *j*) ∈ *S* yields an additional super-additive bonus proportional to the smaller of the two partners' exposures, weighted by *w*<sub>ij</sub>. The portfolio objective is therefore **quadratic**:

> *V*(*x*) = Σ<sub>i</sub> *x*<sub>i</sub> · ELR(*c*<sub>i</sub>) + Σ<sub>(i,j)∈S</sub> *w*<sub>ij</sub> · *x*<sub>i</sub> · *x*<sub>j</sub> · min(*E*<sub>i</sub>, *E*<sub>j</sub>),  (2)

maximized subject to Σ<sub>i</sub> *k*<sub>i</sub> *x*<sub>i</sub> ≤ *B*, where *E*<sub>i</sub> = *a*<sub>i</sub> *p*<sub>i</sub>. Problem (2) is a quadratic knapsack — NP-hard and non-submodular — so value-density greedy selection has no approximation guarantee, motivating global search.

### 3.2 Layer 1 — Prediction

Mitigation effectiveness *m*<sub>i</sub> is estimated from a feature vector (asset criticality, threat velocity, exposure, maturity gap, data sensitivity) using an L2-regularized logistic regressor trained by gradient descent. The learned estimate *m̂*<sub>i</sub> replaces the unknown *m*<sub>i</sub> in Eq. (1).

### 3.3 Layer 2 — Knowledge

Some value is invisible to a model of *marginal* features: crown-jewel assets that are simultaneously under active, high-velocity threat realize an interaction premium. Three expert rules (regulatory exposure, active-threat urgency, maturity-gap opportunity) are encoded as trapezoidal fuzzy memberships and aggregated into a bounded multiplicative weight γ<sub>i</sub> ∈ [1.0, 1.6] that amplifies *m̂*<sub>i</sub>, recovering the premium the predictor cannot observe.

### 3.4 Layer 3 — Quantum-inspired optimization

Each decision *x*<sub>i</sub> is represented by a qubit with phase angle θ<sub>i</sub>; a measurement collapses the qubit to *x*<sub>i</sub> = 1 with probability sin²θ<sub>i</sub> and to 0 otherwise. The population is initialized in uniform superposition (θ<sub>i</sub> = π/4, so sin²θ<sub>i</sub> = 0.5), giving unbiased exploration. In each generation the algorithm (i) measures a population of *P* candidate portfolios, (ii) repairs any budget violation by removing the lowest value-density funded controls, (iii) evaluates the full quadratic objective *V*(*x*) in Eq. (2), and (iv) applies a *rotation gate* that nudges every qubit toward the incumbent best solution by a fixed step Δθ, emulating quantum amplitude amplification. After the generational loop, a 1-flip/swap local-search step refines the collapsed state. Because the fitness in step (iii) is the full quadratic objective, the QIEA is the only method in our comparison that can perceive and exploit control synergies. The procedure is given in Algorithm 1; the parameter settings used throughout (*P* = 24, *G* = 140 generations, Δθ = 0.035 rad) are fixed across all experiments.

> **Algorithm 1.** Quantum-Inspired Evolutionary Algorithm (QIEA) for synergy-aware control selection.
>
> **Input:** controls with costs *k*, exposures *E*, linear values ELR; budget *B*; objective *V*(·); population *P*; generations *G*; rotation step Δθ.
> **Output:** best portfolio *x*\*.
> 1. Initialize qubit angles θ<sub>i</sub> ← π/4 for all *i*; *x*\* ← ∅, *V*\* ← 0.
> 2. **for** *g* = 1 … *G* **do**
> 3.   **for** *p* = 1 … *P* **do**
> 4.     measure: *x*<sub>i</sub> ← 1 with prob. sin²θ<sub>i</sub>, else 0, for all *i*.
> 5.     *x* ← Repair(*x*, *B*)  ▷ drop lowest ELR/cost items until Σ*k*<sub>i</sub>*x*<sub>i</sub> ≤ *B*.
> 6.     **if** *V*(*x*) > *V*\* **then** *x*\* ← *x*, *V*\* ← *V*(*x*).
> 7.   **for** each bit *i* **do**  ▷ rotation-gate update toward incumbent.
> 8.     **if** *x*\*<sub>i</sub> = 1 **then** θ<sub>i</sub> ← min(π/2, θ<sub>i</sub> + Δθ) **else** θ<sub>i</sub> ← max(0, θ<sub>i</sub> − Δθ).
> 9. *x*\* ← LocalSearch(*x*\*, *B*)  ▷ 1-flip / swap hill-climbing on *V*.
> 10. **return** *x*\*.

## 4. Experiments

### 4.1 Setup

We generate a reproducible portfolio of *n* = 120 controls with a known ground truth (fixed seed), a budget *B* equal to 35% of full portfolio cost, and eight complementary synergy pairs; several synergy partners are deliberately low value-density so that myopic methods skip them. The prediction layer attains a mitigation mean-absolute-error of 0.062. We compare the QIEA against four baselines: (i) random search (3200 budget-feasible samples per run); (ii) value-density greedy selection; (iii) a classical genetic algorithm (GA) with population 24, 140 generations, elitist selection (top 20%), single-point crossover, and 3% bit-flip mutation; and (iv) an exact dynamic-programming optimizer of the *linear* surrogate (synergy-blind). For a controlled comparison the GA and the QIEA use the same population size (24) and generation budget (140); the QIEA additionally uses a rotation step Δθ = 0.035 rad. All optimizers maximize the identical quadratic objective of Eq. (2) and all results are means over twenty independent seeds. Figures are generated directly from the experiment.

### 4.2 Results

Table 1 reports the comparison. The QIEA achieves the smallest mean gap to the best solution found (0.06%), ahead of the GA (0.54%), greedy value-density selection (1.84%), and the synergy-blind linear optimizer (2.58%). Random search is not competitive (26.83%), confirming that the search space is non-trivial. The Linear-DP result is the key finding: an *exact* optimizer of the conventional additive objective is outperformed by every interaction-aware heuristic, because its provably optimal additive solution leaves defence-in-depth value unclaimed. This confirms that **synergy modelling, not optimizer choice alone, governs allocation quality**. The QIEA also delivers the highest return on security investment (144.2%) and the fullest budget utilization (100.0%). Although the QIEA and the GA share the same population and generation budget, the QIEA's edge (0.06% versus 0.54% gap) arises from its probabilistic qubit representation: rather than recombining fixed bit-strings, it maintains a per-control probability amplitude that the rotation gate steers toward promising regions, which — combined with the final local-search refinement — yields more reliable exploitation of the sparse synergy pairs. As shown in Fig. 1, the QIEA converges rapidly and stabilizes above the classical GA trajectory and both synergy-blind reference lines within roughly 60 generations.

**Table 1.** Method comparison on the synergy-aware objective, reported as means ± standard deviation over 20 independent random seeds. All optimizers maximize the same quadratic objective *V*(*x*) of Eq. (2) under an identical 35%-of-portfolio budget. "Gap to best" is the relative shortfall against the best solution found by any method on each seed (lower is better); "True risk reduction" uses ground-truth mitigation; "ROI" is (true risk reduction − spend)/spend. Currency figures are in thousands of monetary units. Best values in each column are shown in bold.

| Method | Controls funded | Budget util. | Objective value | True risk reduction | ROI | Gap to best (%) | Runtime (ms) |
|---|---|---|---|---|---|---|---|
| Random Search | 43 | 99.3% | 6669.1 | 6524.4 | 79.4% | 26.828 ± 1.445 | 281.7 |
| Greedy (value-density) | 47 | 99.8% | 8946.0 | 8776.2 | 140.0% | 1.845 ± 0.204 | 0.0 |
| Classical GA | 46 | 99.8% | 9065.1 | 8893.2 | 143.2% | 0.538 ± 0.377 | 136.2 |
| **QIEA (ours)** | 46 | 100.0% | **9109.1** | **8943.1** | **144.2%** | **0.056 ± 0.142** | 251.5 |
| Linear-DP (synergy-blind) | 47 | 100.0% | 8879.4 | 8707.7 | 137.7% | 2.575 ± 0.202 | 29.5 |

Table 2 isolates each layer's contribution through a controlled ablation in which components are enabled cumulatively. Starting from an optimization-only baseline, adding the prediction layer lifts realized annual risk reduction by 2.6%; adding the knowledge layer adds a further 1.1 percentage points (to 3.7%); and adding the synergy-aware objective adds a further 3.8 points, for a total improvement of 7.5% and a rise in return on security investment from 128.5% to 145.3%. The monotonic improvement across all four configurations demonstrates that every layer is justified and that the largest single contribution comes from modelling synergies. Consistent with this, Fig. 2 shows the QIEA dominating the budget–risk-reduction efficiency frontier at every budget level from 10% to 50% of full portfolio cost, with the margin over the greedy and random baselines widening as the budget — and therefore the number of exploitable synergy pairs — grows.

**Table 2.** Layer-wise ablation of the hybrid framework, with components enabled cumulatively (single representative seed, seed offset +99). "Lift vs. base" is the relative change in realized annual risk reduction against the optimization-only configuration. ELR utility is the knowledge-adjusted expected loss reduction; true risk reduction uses ground-truth mitigation. Currency figures are in thousands of monetary units.

| Framework configuration | Controls funded | ELR utility | True risk reduction | ROI | Lift vs. base |
|---|---|---|---|---|---|
| QIEA only (no learning / knowledge / synergy) | 47 | 8057.7 | 8358.3 | 128.5% | +0.0% |
| + Prediction layer | 46 | 8291.4 | 8578.5 | 134.2% | +2.6% |
| + Knowledge layer | 47 | 8828.4 | 8668.5 | 137.2% | +3.7% |
| + Synergy layer (full framework) | 47 | 9137.4 | 8983.3 | 145.3% | +7.5% |

---

**Figure captions**

**Fig. 1.** Convergence of the quantum-inspired optimizer. The curves show the best synergy-aware objective value *V*(*x*) (in thousands) attained by the QIEA and the classical genetic algorithm as a function of generation, with the value-density greedy solution and the synergy-blind linear-DP optimum shown as horizontal reference lines. The QIEA overtakes both references and the GA within approximately 60 generations and remains stable thereafter. (Vector artwork; `artifacts/figure1_convergence.svg`.)

**Fig. 2.** Budget–risk-reduction efficiency frontier. Realized annual risk reduction (in thousands) is plotted against the security budget, expressed as a percentage of full portfolio cost, for the QIEA, classical GA, greedy, and random-search methods. The QIEA Pareto-dominates all baselines across the full budget range, and its advantage widens as the budget admits more exploitable defence-in-depth synergies. (Vector artwork; `artifacts/figure2_frontier.svg`.)

## 5. Discussion and conclusion

### 5.1 Principal findings

We showed that cybersecurity investment prioritization is best treated as a quadratic, synergy-aware optimization problem embedded in a decision-intelligence pipeline that learns control effectiveness, injects expert knowledge, and searches globally with a quantum-inspired optimizer. Two findings stand out. First, the complete framework reaches a 0.06% mean gap to the best attainable solution while remaining fully reproducible on commodity hardware. Second, and more importantly, the comparison with an *exact* synergy-blind optimizer (Table 1) demonstrates that the dominant lever is the **problem formulation**, not the optimizer: once defence-in-depth complementarities are modelled, even simple interaction-aware heuristics beat a provably optimal additive solver. This reframes the practical question from "which optimizer should we use?" to "are we optimizing the right objective?"

### 5.2 Managerial implications

For security leaders, the results carry three implications. (i) Budgeting should target *complementary control bundles* — for example, pairing endpoint detection with centralized analytics, or identity hardening with privileged-access management — rather than independently top-ranked controls. (ii) The framework's transparency supports governance: each recommendation decomposes into a learned effectiveness, an explicit knowledge weight, and a named synergy contribution, so a CISO can justify every funded item to auditors and boards. (iii) The gains are economically material — a 7.5% increase in realized risk reduction and a return on security investment rising to 145% — achieved purely by reallocating a fixed budget, requiring no additional spend.

### 5.3 Limitations and future work

Three limitations bound these conclusions. First, the evaluation uses a synthetic, ground-truth-calibrated portfolio; although this enables exact optimality measurement, validation on real incident and control-efficacy datasets is needed, and is our immediate next step. Second, the formulation is single-objective (expected loss reduction); operational settings also weigh compliance, availability, and implementation risk, motivating a multi-objective extension. Third, the optimizer gap, while small, is non-zero, and the synergy structure is specified by domain experts rather than learned. Future work will (a) validate on empirical datasets, (b) learn synergy weights directly from incident co-occurrence data, (c) extend to multi-objective and robust formulations, and (d) benchmark the quantum-inspired core against true quantum-annealing and gate-based hardware. All code, data generators, figures, and tables are released openly [24] to support independent replication and extension.

### 5.4 Conclusion

This short communication introduced a hybrid decision-intelligence framework that unifies data-driven prediction, knowledge-based reasoning, and quantum-inspired optimization for cybersecurity investment prioritization. By formulating the problem as a synergy-aware quadratic knapsack and solving it with a transparent QIEA, the framework delivers near-optimal, auditable, and economically significant budget allocations, and it establishes that modelling control interactions is the decisive factor for effective security budgeting.

## Declaration of competing interests

The author declares no competing interests.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT author statement

**[Author Name]:** Conceptualization, Methodology, Software, Formal analysis, Investigation, Writing – original draft, Writing – review and editing, Visualization.

## Data availability

The study uses a fully reproducible synthetic dataset generated by the released software (fixed random seed). All code, generated data, figures, and tables are available in the public repository cited as software reference [24].

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used a generative AI assistant to help draft and structure the manuscript and the accompanying open-source code. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## References

[1] L.A. Gordon, M.P. Loeb, The economics of information security investment, ACM Trans. Inf. Syst. Secur. 5 (2002) 438–457. https://doi.org/10.1145/581271.581274.

[2] J. Grossklags, N. Christin, J. Chuang, Secure or insure? A game-theoretic analysis of information security games, in: Proc. 17th Int. Conf. World Wide Web, ACM, 2008, pp. 209–218. https://doi.org/10.1145/1367497.1367526.

[3] V.M. Bier, S. Oliveros, L. Samuelson, Choosing what to protect: strategic defensive allocation against an unknown attacker, J. Public Econ. Theory 9 (2007) 563–587. https://doi.org/10.1111/j.1467-9779.2007.00320.x.

[4] A. Mohammadi, M. Pavlidis, H. Mouratidis, A knowledge-based approach for information security risk assessment, Knowl.-Based Syst. 210 (2020) 106448. https://doi.org/10.1016/j.knosys.2020.106448.

[5] K.-H. Han, J.-H. Kim, Quantum-inspired evolutionary algorithm for a class of combinatorial optimization, IEEE Trans. Evol. Comput. 6 (2002) 580–593. https://doi.org/10.1109/TEVC.2002.804320.

[6] Y. Zhang, S. Wang, G. Ji, A comprehensive survey on quantum-inspired optimization algorithms, Artif. Intell. Rev. 54 (2021) 1–45. https://doi.org/10.1007/s10462-020-09922-6.

[7] A. Layeb, A novel quantum-inspired cuckoo search algorithm for the 0-1 knapsack problem, Int. J. Bio-Inspired Comput. 3 (2011) 297–305. https://doi.org/10.1504/IJBIC.2011.042260.

[8] R. Kumar, et al., A knowledge-based integrated system of hesitant fuzzy set, AHP and TOPSIS for evaluating security–durability of web applications, IEEE Access 8 (2020) 48870–48885. https://doi.org/10.1109/ACCESS.2020.2978038.

[9] E. Altulaihan, M.A. Almaiah, A. Aljughaiman, Anomaly detection IDS for detecting DoS attacks in IoT networks based on machine learning algorithms, Sensors 24 (2024) 713. https://doi.org/10.3390/s24020713.

[10] M. Almars, F. Alharbi, E. Atlam, I. Mahgoub, Machine learning-based security solutions for IoT networks: a comprehensive survey, Sensors 25 (2025) 3341. https://doi.org/10.3390/s25113341.

[11] E. Gyamfi, et al., Evaluating ensemble learning models for intrusion detection in the Internet of Medical Things, Sensors 24 (2024) 5937. https://doi.org/10.3390/s24185937.

[12] G. Folino, et al., Comprehensive analysis and evaluation of anomalous user activity in web server logs, Sensors 24 (2024) 746. https://doi.org/10.3390/s24030746.

[13] Á. Michelena, et al., Design and evaluation of unsupervised machine learning models for anomaly detection in streaming cybersecurity logs, Mathematics 10 (2022) 4043. https://doi.org/10.3390/math10214043.

[14] L.A. Zadeh, Fuzzy sets, Inf. Control 8 (1965) 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X.

[15] M.T.J. Ansari, et al., A fuzzy-based unified decision-making technique to evaluate security risks: a healthcare perspective, Mathematics 11 (2023) 2554. https://doi.org/10.3390/math11112554.

[16] V. Bolgova, et al., Fuzzy logic and its application in the assessment of information security risk of industrial Internet of Things, Symmetry 15 (2023) 1958. https://doi.org/10.3390/sym15101958.

[17] F. Ullah, et al., An intrusion detection system over the IoT data streams using eXplainable artificial intelligence (XAI), Sensors 25 (2025) 847. https://doi.org/10.3390/s25030847.

[18] S. Teng, et al., A synthetic review on enterprise digital transformation: a bibliometric analysis, Sustainability 16 (2024) 1836. https://doi.org/10.3390/su16051836.

[19] Z. Zhang, et al., MOQEA/D: multi-objective QEA with decomposition mechanism and excellent global search and its application, IEEE Trans. Intell. Transp. Syst. 25 (2024) 12456–12470. https://doi.org/10.1109/TITS.2024.3373510.

[20] A. Dey, S. Bhattacharyya, S. Dey, D. Konar, J. Platoš, V. Snášel, L. Mrsic, P. Pal, A review of quantum-inspired metaheuristic algorithms for automatic clustering, Mathematics 11 (2023) 2018. https://doi.org/10.3390/math11092018.

[21] P.K. Gupta, et al., Portfolio optimization problem: a taxonomic review of solution methodologies, IEEE Access 11 (2023) 33100–33120. https://doi.org/10.1109/ACCESS.2023.3263198.

[22] F. Glover, G. Kochenberger, Y. Du, Quantum bridge analytics I: a tutorial on formulating and using QUBO models, 4OR 17 (2019) 335–371. https://doi.org/10.1007/s10288-019-00424-y.

[23] A. Diaba, et al., Deep anomaly detection framework utilizing federated learning for electricity theft zero-day cyberattacks, Sensors 24 (2024) 3236. https://doi.org/10.3390/s24103236.

[24] [Author Name], QAI-CIP: Quantum-Inspired AI for Cybersecurity Investment Prioritization [software], GitHub, 2026. https://github.com/AshokPrjcts/Journal/tree/main/qai-cyber-investment.
