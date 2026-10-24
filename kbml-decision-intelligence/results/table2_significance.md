| Comparison | Metric | Baseline mean | KG-DI mean | Mean diff. | Cohen's dz | p-value |
|---|---|---|---|---|---|---|
| KG-DI vs Logistic Regression | AUC | 0.683 | 0.693 | +0.010 | +1.00 | 1.1e-06 |
| KG-DI vs Decision Tree | AUC | 0.673 | 0.693 | +0.020 | +0.46 | 0.025 |
| KG-DI vs Random Forest | AUC | 0.682 | 0.693 | +0.011 | +0.71 | 4.8e-04 |
| KG-DI vs KG-DI without knowledge prior (ablation) | AUC | 0.683 | 0.693 | +0.010 | +0.59 | 0.004 |

*Table 2. Paired significance tests across 24 matched splits (two-sided paired t-test; positive mean difference favors KG-DI). KG-DI significantly outperforms the weaker baselines and the ablation confirms the knowledge prior's contribution.*