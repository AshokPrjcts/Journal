# Knowledge-Based Machine Learning for Data-Driven Decision Intelligence in Digital Transformation and Entrepreneurship

Reproducible code accompanying the short communication submitted to
*Knowledge-Based Systems* (Elsevier, ISSN 0950-7051).

This repository implements a **knowledge-guided gradient-boosting** model that
predicts firm-level **digital-transformation readiness** from theory-informed
digital-transformation / entrepreneurship constructs, and converts the model's
probabilistic output into **cost-sensitive managerial decisions** (the
decision-intelligence layer).

> **Zero dependencies.** Everything — data generation, models, metrics, the
> decision layer, and the publication figures (rendered as vector SVG) — is
> written in the **Python standard library only**. No `numpy`, `pandas`,
> `scikit-learn`, or `matplotlib` are required. This maximizes reproducibility.

## Repository structure

```
kbml-decision-intelligence/
├── src/
│   ├── data.py         # Knowledge-based synthetic data generator
│   ├── models.py       # Baselines, KG-DI, metrics, decision layer
│   ├── stats_tests.py  # Paired significance tests (t-test, Cohen's dz)
│   └── experiment.py   # Runs everything; writes figures + tables to results/
├── results/            # Generated artifacts (2 figures + 2 tables + sweep)
├── requirements.txt
├── manuscript.md       # Short-communication manuscript
└── README.md
```

## How to reproduce

```bash
python3 src/experiment.py
```

This regenerates all four result artifacts deterministically (fixed seeds):

| Artifact | File |
|---|---|
| Figure 1 — ROC curves + KG-DI decision landscape | `results/fig1_models.svg` |
| Figure 2 — Factorial ablation + cost-sensitive thresholding | `results/fig2_ablation.svg` |
| Table 1 — Model comparison (24 matched splits) | `results/table1_performance.{csv,md}` |
| Table 2 — Paired significance tests | `results/table2_significance.{csv,md}` |
| Supplementary — ablation + sensitivity sweep | `results/ablation_sweep.csv` |

Runtime: a few minutes on a laptop (pure Python; 24 matched splits).

## Method summary

1. **Knowledge base (`data.py`).** Eight firm-level constructs (digital
   capability, data-driven culture, innovation intensity, leadership support,
   IT investment, workforce skills, organizational agility, entrepreneurial
   risk) with theory-informed weights. The ground-truth readiness label is a
   transparent generative process combining a linear knowledge score with
   nonlinear interaction and threshold effects plus label noise.

2. **Models (`models.py`).**
   - Baselines: Logistic Regression, Decision Tree, Random Forest.
   - **Proposed: KG-DI** — stochastic gradient boosting whose initial score is a
     *knowledge prior* (theory-weighted feature combination), then boosts the
     nonlinear residual. An **ablation** (boosting without the prior) isolates
     the value of injecting domain knowledge.

3. **Decision-intelligence layer (`models.py::DecisionPolicy`).** Converts
   predicted probabilities into decisions by minimizing expected
   misclassification cost under asymmetric FP/FN costs, yielding an optimized
   decision threshold `t*`.

4. **Statistics (`stats_tests.py`).** Paired two-sided *t*-test with Cohen's
   *d_z* effect size for matched per-split comparisons.

## Key results (24 matched holdout splits)

- **KG-DI attains the highest AUC (0.693)**, significantly beating logistic
  regression (*p* = 1.1e-06), random forest (*p* = 4.8e-04), and the decision
  tree (*p* = 0.025).
- The **knowledge prior** contributes a statistically significant gain over the
  boosting-only ablation (*p* = 0.004).
- A 2x2 factorial ablation and a knowledge-scale sensitivity sweep confirm that
  knowledge and boosting are complementary, with the combination best.
- Cost-sensitive thresholding reduces expected managerial misclassification cost
  by about **60%** when missing a ready firm (false negative) is costly.

See `results/` for exact numbers and `manuscript.md` for the full write-up.

## Data & ethics

The dataset is **programmatically generated** from published
digital-transformation / entrepreneurship constructs and is clearly labeled as
synthetic. No real personal or firm data are used. For an empirical submission,
the generator in `data.py` can be swapped for a real survey/firm-level dataset
without changing the modeling or decision pipeline.

## License

MIT (see `LICENSE`).
