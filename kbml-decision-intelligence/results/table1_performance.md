| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.628±0.032 | 0.649±0.045 | 0.757±0.052 | 0.698±0.037 | 0.683±0.035 |
| Decision Tree | 0.592±0.046 | 0.642±0.056 | 0.649±0.085 | 0.642±0.054 | 0.673±0.049 |
| Random Forest | 0.623±0.035 | 0.653±0.046 | 0.725±0.055 | 0.686±0.040 | 0.682±0.032 |
| KG-DI without knowledge prior (ablation) | 0.629±0.031 | 0.656±0.041 | 0.733±0.059 | 0.691±0.039 | 0.683±0.036 |
| KG-DI (proposed) * | 0.638±0.032 | 0.683±0.048 | 0.683±0.040 | 0.682±0.035 | 0.693±0.034 |

*Table 1. Predictive performance over 24 matched holdout splits (mean +/- SD; N=1000 firms/split). The proposed KG-DI (*) attains the highest AUC. The ablation row removes only the knowledge prior, isolating its contribution.*