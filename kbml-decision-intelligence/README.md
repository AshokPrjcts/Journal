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
│   ├── data.py        # Knowledge-based synthetic data generator
│   ├── models.py      # Baselines, Knowledge-Guided GBM, metrics, decision layer
│   └── experiment.py  # Runs everything; writes figures + tables to results/
├── results/           # Generated artifacts (figures + tables)
├── requirements.txt
├── manuscript.md      # Short-communication manuscript
└── README.md
```

## How to reproduce

```bash
python3 src/experiment.py
```

This regenerates all four result artifacts deterministically (fixed seeds):

| Artifact | File |
|---|---|
| Figure 1 — ROC curves (all models) | `results/fig1_roc.svg` |
| Figure 2 — Decision cost curve + knowledge weights | `results/fig2_decision.svg` |
| Table 1 — Model comparison (repeated holdout) | `results/table1_performance.{csv,md}` |
| Table 2 — Cost-sensitive decision analysis | `results/table2_decision.{csv,md}` |

Runtime: ~1–2 minutes on a laptop (pure Python).

## Method summary

1. **Knowledge base (`data.py`).** Eight firm-level constructs (digital
   capability, data-driven culture, innovation intensity, leadership support,
   IT investment, workforce skills, organizational agility, entrepreneurial
   risk) with theory-informed weights. The ground-truth readiness label is a
   transparent generative process combining a linear knowledge score with
   nonlinear interaction and threshold effects plus label noise.

2. **Models (`models.py`).**
   - Baselines: Logistic Regression, Decision Tree, Random Forest.
   - **Proposed: Knowledge-Guided GBM** — stochastic gradient boosting whose
     initial score is a *knowledge prior* (theory-weighted feature combination),
     then boosts the nonlinear residual. An **ablation** (boosting without the
     prior) isolates the value of injecting domain knowledge.

3. **Decision-intelligence layer (`models.py::DecisionPolicy`).** Converts
   predicted probabilities into decisions by minimizing expected
   misclassification cost under asymmetric FP/FN costs, yielding an optimized
   decision threshold `t*`.

## Key results

- The proposed **Knowledge-Based GBM attains the highest AUC** across repeated
  holdout splits, and the knowledge prior produces a measurable lift over the
  ablation.
- Cost-sensitive thresholding reduces expected managerial misclassification
  cost by up to **~57%** when missing a ready firm (false negative) is costly.

See `results/` for exact numbers and `manuscript.md` for the full write-up.

## Data & ethics

The dataset is **programmatically generated** from published
digital-transformation / entrepreneurship constructs and is clearly labeled as
synthetic. No real personal or firm data are used. For an empirical submission,
the generator in `data.py` can be swapped for a real survey/firm-level dataset
without changing the modeling or decision pipeline.

## License

MIT (see `LICENSE`).
