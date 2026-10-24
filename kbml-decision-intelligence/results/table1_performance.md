| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.688±0.018 | 0.703±0.009 | 0.803±0.024 | 0.749±0.015 | 0.734±0.025 |
| Decision Tree (baseline) | 0.630±0.017 | 0.667±0.018 | 0.731±0.039 | 0.697±0.017 | 0.703±0.020 |
| Random Forest (baseline) | 0.667±0.017 | 0.692±0.011 | 0.770±0.025 | 0.729±0.017 | 0.734±0.035 |
| GBM without knowledge prior (ablation) | 0.672±0.030 | 0.701±0.020 | 0.761±0.035 | 0.730±0.026 | 0.732±0.032 |
| Knowledge-Based GBM (ours) | 0.670±0.026 | 0.698±0.012 | 0.763±0.041 | 0.729±0.025 | 0.737±0.035 |

*Table 1. Classification performance (mean +/- SD over 5 repeated holdout splits). The proposed Knowledge-Based GBM attains the highest AUC; the ablation row isolates the contribution of the knowledge prior.*