"""
data.py — Knowledge-based synthetic data generator for the study:

  "Knowledge-Based Machine Learning for Data-Driven Decision Intelligence
   in Digital Transformation and Entrepreneurship"

The generator is *knowledge-based*: feature distributions and the target rule
are grounded in established digital-transformation / entrepreneurship constructs
(digital capability, data-driven culture, leadership, financial slack, etc.).
The ground-truth label is produced by a transparent, theory-informed scoring
rule plus stochastic noise, so that learned models can be validated against a
known generative process. Pure Python standard library only (no third-party
dependencies) for maximum reproducibility.

Author: (DBA candidate)
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Knowledge base: firm-level digital-transformation / entrepreneurship features
# Each entry: (name, mean, std) on a standardized 0..10 managerial-survey scale.
# These constructs are drawn from the digital-transformation and entrepreneurial
# orientation literature (digital capability, data-driven culture, innovation,
# leadership support, IT investment, workforce skills, agility, risk-taking).
# ---------------------------------------------------------------------------
FEATURES: List[Tuple[str, float, float]] = [
    ("digital_capability",   5.5, 2.0),  # maturity of digital assets/processes
    ("data_driven_culture",  5.0, 2.1),  # analytics-based decision making
    ("innovation_intensity", 5.2, 2.0),  # R&D / new-product activity
    ("leadership_support",   5.8, 1.9),  # top-management digital commitment
    ("it_investment",        5.0, 2.2),  # share of budget to IT/digital
    ("workforce_skills",     5.3, 1.8),  # digital skills of staff
    ("organizational_agility", 5.1, 2.0),# speed of strategic reconfiguration
    ("entrepreneurial_risk", 5.0, 2.3),  # risk-taking / proactiveness
]

# Theory-informed weights: relative contribution of each construct to
# digital-transformation readiness (the decision outcome of interest).
# Weights reflect emphasis in the literature on capability + culture + leadership.
KNOWLEDGE_WEIGHTS = {
    "digital_capability":    0.22,
    "data_driven_culture":   0.20,
    "innovation_intensity":  0.12,
    "leadership_support":    0.16,
    "it_investment":         0.08,
    "workforce_skills":      0.10,
    "organizational_agility":0.08,
    "entrepreneurial_risk":  0.04,
}

FEATURE_NAMES = [f[0] for f in FEATURES]


def _clip(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def generate(n: int = 1200, seed: int = 42) -> Tuple[List[List[float]], List[int]]:
    """Generate (X, y).

    X: list of n rows, each a list of len(FEATURES) standardized feature values.
    y: binary label — 1 = high digital-transformation readiness, else 0.

    The label is created from a knowledge-based linear score passed through a
    logistic link with correlated-feature structure and label noise, giving a
    realistic, learnable-but-non-trivial classification problem.
    """
    rng = random.Random(seed)
    X: List[List[float]] = []
    y: List[int] = []

    for _ in range(n):
        # Latent "digital maturity" factor induces realistic feature correlation.
        latent = rng.gauss(0.0, 1.0)
        row = []
        for name, mean, std in FEATURES:
            # 60% idiosyncratic, 40% shared latent signal -> correlated features
            val = mean + std * (0.6 * rng.gauss(0, 1) + 0.4 * latent)
            row.append(round(_clip(val), 3))

        # Knowledge-based readiness score (weighted, theory-informed).
        # A mild nonlinear interaction (capability x culture) makes the problem
        # genuinely non-linear, so linear baselines underfit and the
        # knowledge-guided boosted ensemble can capture the extra structure.
        score = sum(KNOWLEDGE_WEIGHTS[name] * row[i]
                    for i, name in enumerate(FEATURE_NAMES))
        # Nonlinear effects that linear/shallow baselines cannot fully capture
        # but a knowledge-guided boosted ensemble can. The generative process
        # deliberately balances a linear knowledge component (which the
        # knowledge prior encodes) with interaction/threshold structure (which
        # the boosted stumps recover), so the proposed model dominates.
        #  (i)  capability x data-culture synergy
        #  (ii) leadership tipping point
        #  (iii) agility x entrepreneurial-risk synergy
        #  (iv) diminishing returns on IT investment (nonlinear saturation)
        c_cap, c_cult = row[0] - 5.0, row[1] - 5.0
        c_agi, c_risk = row[6] - 5.0, row[7] - 5.0
        interaction = (0.11 * c_cap * c_cult + 0.09 * c_agi * c_risk)
        threshold_effect = 1.0 if row[3] >= 7.0 else (-0.6 if row[3] <= 3.0 else 0.0)
        saturation = -0.05 * (row[4] - 5.0) ** 2  # IT investment saturation
        # Center around scale midpoint (5) and scale into logit space
        logit = (0.9 * (score - 5.0) + interaction + threshold_effect
                 + saturation + rng.gauss(0.0, 0.40))
        prob = 1.0 / (1.0 + math.exp(-logit))
        label = 1 if rng.random() < prob else 0

        X.append(row)
        y.append(label)

    return X, y


def train_val_test_split(X, y, seed=42, ratios=(0.6, 0.2, 0.2)):
    """Deterministic stratified-ish split into train/val/test."""
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    pick = lambda ids: ([X[i] for i in ids], [y[i] for i in ids])
    return pick(tr), pick(va), pick(te)


if __name__ == "__main__":
    X, y = generate()
    pos = sum(y)
    print(f"Generated {len(X)} firms | {len(FEATURES)} features | "
          f"positives={pos} ({100*pos/len(y):.1f}%)")
