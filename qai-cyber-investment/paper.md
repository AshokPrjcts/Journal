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

Chief information security officers must allocate finite budgets across competing security controls, yet prevailing practice relies on ranked heuristics that ignore interactions between controls. We present a hybrid *decision-intelligence* framework for cybersecurity investment prioritization that couples three cooperating layers: (i) a data-driven prediction layer that learns per-control breach-mitigation effectiveness; (ii) a knowledge layer that encodes expert security rules through fuzzy amplification to recover value the predictor cannot observe from marginal features; and (iii) a quantum-inspired evolutionary optimizer (QIEA) that solves the budget-constrained selection problem. Because co-funding complementary controls yields super-additive "defence-in-depth" protection, the allocation problem is quadratic and non-submodular, which defeats myopic value-density heuristics. On a reproducible 120-control portfolio averaged over twenty independent seeds, the full framework attains a mean optimality gap of 0.06% to the best solution found, compared with 0.54% for a classical genetic algorithm, 1.84% for value-density greedy selection, and 2.58% for an exact linear optimizer that is blind to synergies. A layer-wise ablation confirms that each component contributes positively, improving realized annual risk reduction by 7.5% over an optimization-only baseline and raising return on security investment to 144%. The framework is implemented in dependency-free Python and released openly, offering security managers a transparent, auditable decision-support tool. Results indicate that modelling control synergies, rather than optimizer choice alone, is the decisive factor for effective cybersecurity budgeting.

**Keywords:** cybersecurity; investment; quantum-inspired; optimization; knowledge; decision-support; prioritization

---

## 1. Introduction

Organizations face an expanding threat surface under hard budget constraints, forcing security leaders to decide *which* controls to fund rather than *whether* to invest [1,2]. In practice these decisions are dominated by qualitative risk matrices and single-criterion rankings that treat each control independently [3]. This independence assumption is unsafe: security controls interact. Endpoint detection is far more valuable alongside centralized log analytics; identity hardening compounds with privileged-access management. Such "defence-in-depth" complementarities make the true value of a portfolio *super-additive*, so a ranking that scores controls in isolation systematically misallocates capital [4].

Three capabilities are therefore required of a credible decision-support system: it must (i) *estimate* how effective each control is from data, (ii) *incorporate expert knowledge* that the data alone cannot express, and (iii) *optimize* the joint, interaction-aware objective under a budget. Each capability maps onto a mature strand of artificial intelligence — machine learning, knowledge-based systems, and computational-intelligence optimization — yet they are rarely combined in the cybersecurity-investment setting.

We contribute a *hybrid decision-intelligence framework* that unifies the three. Its optimization core is a **quantum-inspired evolutionary algorithm (QIEA)**: a classical algorithm that represents each funding decision as a qubit in superposition and uses a rotation-gate update to emulate amplitude amplification, giving strong global search on commodity hardware without quantum computers [5,6]. Our specific contributions are:

1. A three-layer architecture (prediction → knowledge → quantum-inspired optimization) that converts raw threat data into auditable investment recommendations (Section 3).
2. A formulation of cybersecurity budgeting as a **quadratic, synergy-aware** knapsack problem that captures defence-in-depth, and a QIEA that solves it (Sections 3–4).
3. A reproducible evaluation showing the framework closes the optimality gap to 0.06% and that modelling synergies — not the optimizer alone — drives performance (Section 4).

## 2. Related work

**Security economics and investment.** Gordon and Loeb established the economic foundations of security investment [1], followed by portfolio and game-theoretic treatments [2,4]. These works largely assume additive utilities and do not provide an operational, learning-enabled decision tool.

**Knowledge-based decision support.** Expert systems and fuzzy inference have long supported risk assessment under uncertainty [3,7], and remain central to the scope of knowledge-based systems. We adopt fuzzy amplification as the mechanism by which domain rules correct data-driven estimates.

**Quantum-inspired optimization.** Han and Kim introduced quantum-inspired evolutionary algorithms for combinatorial problems [5], with subsequent advances in knapsack and feature-selection tasks [6,8]. We are, to our knowledge, the first to apply a QIEA to synergy-aware cybersecurity investment prioritization, and to combine it with a learning and knowledge layer in a single framework.

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

Each decision *x*<sub>i</sub> is a qubit with angle θ<sub>i</sub>; measurement sets *x*<sub>i</sub> = 1 with probability sin²θ<sub>i</sub>, so the population begins in uniform superposition (θ = π/4). Each generation measures a population, repairs budget violations, and rotates every qubit toward the best solution found (rotation step Δθ), emulating amplitude amplification. A final 1-flip/swap local-search step refines the collapsed state. The optimizer evaluates the full quadratic objective (2), so it alone captures synergies.

## 4. Experiments

### 4.1 Setup

We generate a reproducible portfolio of *n* = 120 controls with a known ground truth (fixed seed), a budget *B* equal to 35% of full portfolio cost, and eight complementary synergy pairs; several synergy partners are deliberately low value-density so that myopic methods skip them. The prediction layer attains a mitigation mean-absolute-error of 0.062. We compare the QIEA against random search, value-density greedy selection, a classical genetic algorithm (GA), and an exact dynamic-programming optimizer of the *linear* surrogate (synergy-blind). All results are means over twenty independent seeds. Figures are generated directly from the experiment.

### 4.2 Results

Table 1 reports the comparison. The QIEA achieves the smallest mean gap to the best solution found (0.06%), ahead of the GA (0.54%), greedy (1.84%), and the synergy-blind linear optimizer (2.58%). The last result is the key finding: an *exact* optimizer of the conventional additive objective is outperformed by a heuristic that reasons over interactions, confirming that **synergy modelling, not optimizer choice alone, governs allocation quality**. Figure 1 shows the QIEA converging smoothly above the GA and the synergy-blind references.

**Table 1.** Method comparison on the synergy-aware objective (means over 20 seeds). Gap is relative to the best solution found by any method; lower is better.

| Method | Controls funded | Budget util. | Objective value | True risk reduction | ROI | Gap to best (%) | Runtime (ms) |
|---|---|---|---|---|---|---|---|
| Random Search | 43 | 99.3% | 6669.1 | 6524.4 | 79.4% | 26.828 ± 1.445 | 281.7 |
| Greedy (value-density) | 47 | 99.8% | 8946.0 | 8776.2 | 140.0% | 1.845 ± 0.204 | 0.0 |
| Classical GA | 46 | 99.8% | 9065.1 | 8893.2 | 143.2% | 0.538 ± 0.377 | 136.2 |
| **QIEA (ours)** | 46 | 100.0% | **9109.1** | **8943.1** | **144.2%** | **0.056 ± 0.142** | 251.5 |
| Linear-DP (synergy-blind) | 47 | 100.0% | 8879.4 | 8707.7 | 137.7% | 2.575 ± 0.202 | 29.5 |

Table 2 isolates each layer's contribution. Starting from an optimization-only baseline, adding the prediction layer lifts realized risk reduction by 2.6%, the knowledge layer by a further 1.1 percentage points (to 3.7%), and the synergy-aware objective by a further 3.8 points (to 7.5% overall), with return on security investment rising from 128% to 145%. Every layer is justified. Figure 2 shows that the QIEA dominates the budget–risk-reduction efficiency frontier across budget levels from 10% to 50% of portfolio cost.

**Table 2.** Layer-wise ablation (single representative seed). Lift is relative to the optimization-only base configuration.

| Framework configuration | Controls funded | ELR utility | True risk reduction | ROI | Lift vs. base |
|---|---|---|---|---|---|
| QIEA only (no learning / knowledge / synergy) | 47 | 8057.7 | 8358.3 | 128.5% | +0.0% |
| + Prediction layer | 46 | 8291.4 | 8578.5 | 134.2% | +2.6% |
| + Knowledge layer | 47 | 8828.4 | 8668.5 | 137.2% | +3.7% |
| + Synergy layer (full framework) | 47 | 9137.4 | 8983.3 | 145.3% | +7.5% |

## 5. Discussion and conclusion

We showed that cybersecurity investment prioritization is best treated as a quadratic, synergy-aware optimization problem embedded in a decision-intelligence pipeline that learns effectiveness, injects expert knowledge, and searches globally with a quantum-inspired optimizer. The framework is transparent and auditable — each recommendation decomposes into a predicted effectiveness, a knowledge weight, and a synergy contribution — which matters for security governance. The principal managerial implication is that organizations should fund *complementary bundles* rather than top-ranked controls in isolation.

Limitations include reliance on a synthetic (though ground-truth-calibrated) portfolio and a single-objective formulation; the optimizer gap, while small, is not zero. Future work will validate on incident datasets, extend to multi-objective settings (e.g., risk versus compliance), and benchmark against true quantum hardware. All code and artifacts are released to support replication.

## Declaration of competing interests

The author declares no competing interests.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT author statement

**[Author Name]:** Conceptualization, Methodology, Software, Formal analysis, Investigation, Writing – original draft, Writing – review and editing, Visualization.

## Data availability

The study uses a fully reproducible synthetic dataset generated by the released software (fixed random seed). All code, generated data, figures, and tables are available in the public repository cited as software reference [9].

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used a generative AI assistant to help draft and structure the manuscript and the accompanying open-source code. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## References

[1] L.A. Gordon, M.P. Loeb, The economics of information security investment, ACM Trans. Inf. Syst. Secur. 5 (2002) 438–457. https://doi.org/10.1145/581271.581274.

[2] J. Grossklags, N. Christin, J. Chuang, Secure or insure? A game-theoretic analysis of information security games, in: Proc. 17th Int. Conf. World Wide Web, ACM, 2008, pp. 209–218. https://doi.org/10.1145/1367497.1367526.

[3] A. Mohammadi, M. Pavlidis, H. Mouratidis, A knowledge-based approach for information security risk assessment, Knowl.-Based Syst. 210 (2020) 106448. https://doi.org/10.1016/j.knosys.2020.106448.

[4] V.M. Bier, S. Oliveros, L. Samuelson, Choosing what to protect: strategic defensive allocation against an unknown attacker, J. Public Econ. Theory 9 (2007) 563–587. https://doi.org/10.1111/j.1467-9779.2007.00320.x.

[5] K.-H. Han, J.-H. Kim, Quantum-inspired evolutionary algorithm for a class of combinatorial optimization, IEEE Trans. Evol. Comput. 6 (2002) 580–593. https://doi.org/10.1109/TEVC.2002.804320.

[6] Y. Zhang, S. Wang, G. Ji, A comprehensive survey on quantum-inspired optimization algorithms, Artif. Intell. Rev. 54 (2021) 1–45. https://doi.org/10.1007/s10462-020-09922-6.

[7] L.A. Zadeh, Fuzzy sets, Inf. Control 8 (1965) 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X.

[8] A. Layeb, A novel quantum-inspired cuckoo search algorithm for the 0-1 knapsack problem, Int. J. Bio-Inspired Comput. 3 (2011) 297–305. https://doi.org/10.1504/IJBIC.2011.042260.

[9] [Author Name], QAI-CIP: Quantum-Inspired AI for Cybersecurity Investment Prioritization [software], GitHub, 2026. https://github.com/AshokPrjcts/Journal/tree/main/qai-cyber-investment.
