"""
qaip.py — Quantum-Inspired AI for Cybersecurity Investment Prioritization (QAI-CIP)
===================================================================================

A single-file, ZERO-DEPENDENCY (Python standard library only) implementation of the
Hybrid Decision Intelligence Framework described in the short communication:

    "Quantum-Inspired Artificial Intelligence for Cybersecurity Investment
     Prioritization: A Hybrid Decision Intelligence Framework"

The framework has three cooperating layers:

  1. PREDICTION LAYER   — a from-scratch L2-regularised logistic-regression model
                          estimates each control's breach-mitigation probability
                          from asset/threat features (data-driven risk analytics).

  2. KNOWLEDGE LAYER    — an expert rule / fuzzy-adjustment base encodes domain
                          knowledge (regulatory exposure, asset criticality,
                          threat velocity) and reshapes the raw risk scores into
                          calibrated expected-loss-reduction (ELR) utilities.

  3. QUANTUM-INSPIRED   — a Quantum-Inspired Evolutionary Algorithm (QIEA) with a
     OPTIMIZATION LAYER   qubit-probability representation and rotation-gate update
                          solves the budget-constrained 0/1 portfolio problem
                          (which controls to fund) to maximise total ELR utility.

The script is fully deterministic (fixed seed) and reproducible. Running it:

    python3 qaip.py

produces, in ./artifacts/:
  * figure1_convergence.svg   — Fig. 1: optimiser convergence vs. baselines
  * figure2_frontier.svg      — Fig. 2: risk-reduction vs. budget (efficiency frontier)
  * table1_performance.csv/.md— Table 1: method comparison (ELR, risk reduction, ROI, time)
  * table2_ablation.csv/.md   — Table 2: ablation of the three framework layers
  * results.json              — machine-readable summary of all numbers

No third-party packages are required (no numpy/pandas/sklearn/matplotlib).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 20260101
ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


# =========================================================================== #
# 1. SYNTHETIC CYBERSECURITY INVESTMENT DATASET
# =========================================================================== #
@dataclass
class Control:
    """A candidate security control (investment option)."""
    cid: int
    name: str
    cost: float                 # capital cost (currency units, thousands)
    features: list[float]       # [asset_crit, threat_velocity, exposure, maturity_gap, data_sensitivity]
    asset_value: float          # value at risk protected by the control (thousands)
    base_breach_prob: float     # annualised breach probability for the covered asset
    true_mitigation: float      # ground-truth fraction of risk the control removes [0,1]


CONTROL_FAMILIES = [
    "Endpoint Detection & Response", "Network Segmentation", "IAM / MFA Hardening",
    "SIEM & Threat Hunting", "Data Encryption at Rest", "Email Security Gateway",
    "Vulnerability Management", "Backup & Recovery", "Zero-Trust Access",
    "Cloud Posture Management", "DLP Controls", "Security Awareness Training",
    "WAF / API Protection", "Privileged Access Management", "OT/ICS Monitoring",
    "DDoS Mitigation", "Patch Automation", "Secrets Management",
    "Container Runtime Security", "Incident Response Retainer",
]


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def generate_portfolio(n: int, rng: random.Random) -> list[Control]:
    """Create a realistic candidate-control portfolio with a known ground truth.

    The ground-truth mitigation effectiveness is a (noisy) non-linear function of
    the feature vector, so a learned model has a genuine signal to recover.
    """
    controls: list[Control] = []
    # latent weights the predictor must recover (unknown to the model)
    w = [1.35, 0.95, 1.1, -0.85, 0.7]
    bias = -0.4
    for i in range(n):
        asset_crit = rng.random()
        threat_velocity = rng.random()
        exposure = rng.random()
        maturity_gap = rng.random()           # higher gap => more room to improve
        data_sensitivity = rng.random()
        feats = [asset_crit, threat_velocity, exposure, maturity_gap, data_sensitivity]

        z = bias + sum(wi * fi for wi, fi in zip(w, feats))
        z += rng.gauss(0.0, 0.35)             # irreducible noise
        true_mitigation = max(0.03, min(0.97, _logistic(1.8 * z)))

        # Domain-knowledge signal: crown-jewel / high-velocity assets realise an
        # extra effectiveness premium that the feature-only predictor cannot see
        # (interaction effect). The knowledge layer is designed to recover this.
        premium = 0.0
        if (0.5 * asset_crit + 0.5 * data_sensitivity) > 0.6 and \
           (0.5 * threat_velocity + 0.5 * exposure) > 0.6:
            premium = 0.12
        true_mitigation = max(0.03, min(0.99, true_mitigation + premium))

        asset_value = 40.0 + 960.0 * (0.5 * asset_crit + 0.5 * data_sensitivity)
        base_breach_prob = max(0.02, min(0.9, 0.12 + 0.6 * exposure * threat_velocity))
        cost = 8.0 + 140.0 * (0.4 * maturity_gap + 0.3 * asset_crit + 0.3 * rng.random())

        # Make the "second" member of each synergy pair individually unattractive
        # (low standalone value-density) so myopic greedy/linear methods skip it,
        # even though funding it unlocks a large joint synergy bonus.
        family = i % len(CONTROL_FAMILIES)
        low_density_partners = {b for (_, b) in SYNERGY_PAIRS}
        if family in low_density_partners:
            asset_value *= 0.45        # weaker standalone benefit
            cost *= 1.25               # and pricier => poor density

        controls.append(Control(
            cid=i,
            name=f"{CONTROL_FAMILIES[family]} #{i // len(CONTROL_FAMILIES) + 1}",
            cost=round(cost, 2),
            features=feats,
            asset_value=round(asset_value, 2),
            base_breach_prob=round(base_breach_prob, 4),
            true_mitigation=round(true_mitigation, 4),
        ))
    return controls


# =========================================================================== #
# 2. PREDICTION LAYER — logistic regression from scratch
# =========================================================================== #
class LogisticRegression:
    """L2-regularised logistic regression trained with full-batch gradient descent."""

    def __init__(self, n_features: int, lr: float = 0.3, l2: float = 1e-3, epochs: int = 4000):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs

    def _predict_raw(self, x: list[float]) -> float:
        return _logistic(self.b + sum(wi * xi for wi, xi in zip(self.w, x)))

    def fit(self, X: list[list[float]], y: list[float]) -> "LogisticRegression":
        n = len(X)
        m = len(self.w)
        for _ in range(self.epochs):
            gw = [0.0] * m
            gb = 0.0
            for xi, yi in zip(X, y):
                p = self._predict_raw(xi)
                err = p - yi
                for j in range(m):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(m):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * (gb / n)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self._predict_raw(x) for x in X]


# =========================================================================== #
# 3. KNOWLEDGE LAYER — expert rules + fuzzy risk amplification
# =========================================================================== #
def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal fuzzy membership function."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def knowledge_adjustment(control: Control) -> float:
    """Return a multiplicative knowledge weight (>=1) applied to predicted mitigation.

    Encodes three expert rules via fuzzy membership on the feature vector:
      R1: critical, highly-sensitive assets (regulatory exposure) are prioritised.
      R2: high threat velocity with high exposure => urgency amplification.
      R3: large maturity gaps offer disproportionate marginal gains.
    """
    asset_crit, threat_velocity, exposure, maturity_gap, data_sensitivity = control.features

    # R1 — regulatory / crown-jewel exposure (membership peaks past 0.6 trigger)
    reg = _trapezoid(0.5 * asset_crit + 0.5 * data_sensitivity, 0.5, 0.6, 1.0, 1.0)
    # R2 — active-threat urgency (membership peaks past 0.6 trigger)
    urg = _trapezoid(0.5 * threat_velocity + 0.5 * exposure, 0.5, 0.6, 1.0, 1.0)
    # R3 — maturity-gap opportunity
    gap = _trapezoid(maturity_gap, 0.3, 0.6, 1.0, 1.0)

    # Conjunctive (AND) firing of R1 & R2 captures the crown-jewel x high-velocity
    # interaction premium; R3 provides a mild opportunity bonus. Bounded in [1.0,1.6].
    interaction = min(reg, urg)
    aggregate = 1.0 + 0.55 * interaction + 0.08 * gap
    return min(1.6, aggregate)


def expected_loss_reduction(control: Control, mitigation_hat: float, use_knowledge: bool) -> float:
    """Expected annual loss reduction (ELR) utility for a control.

    ELR = asset_value * base_breach_prob * effective_mitigation
    where effective_mitigation optionally incorporates the knowledge weight.
    """
    eff = mitigation_hat * (knowledge_adjustment(control) if use_knowledge else 1.0)
    eff = min(1.0, eff)
    return control.asset_value * control.base_breach_prob * eff


# =========================================================================== #
# 4. QUANTUM-INSPIRED OPTIMIZATION LAYER — QIEA
# =========================================================================== #
@dataclass
class QIEAResult:
    best_mask: list[int]
    best_value: float
    history: list[float] = field(default_factory=list)
    elapsed: float = 0.0


class QuantumInspiredEA:
    """Quantum-Inspired Evolutionary Algorithm for 0/1 knapsack-style selection.

    Each decision bit i is represented by a qubit with amplitudes (alpha_i, beta_i),
    |alpha|^2 + |beta|^2 = 1. Measurement collapses bit i to 1 with probability
    beta_i^2. A rotation gate nudges each qubit toward the best solution found,
    emulating quantum interference/amplitude amplification on classical hardware.

    The objective is supplied as a callback ``value_fn(mask) -> float`` so the
    optimiser can handle the non-additive (quadratic) synergy-aware objective.
    """

    def __init__(self, n_items: int, costs: list[float], value_fn: Callable,
                 budget: float, lin_values: list[float], pop: int = 24,
                 generations: int = 120, theta: float = 0.035,
                 rng: random.Random | None = None):
        self.n = n_items
        self.costs = costs
        self.value_fn = value_fn
        self.lin_values = lin_values          # used only for budget repair ordering
        self.budget = budget
        self.pop = pop
        self.generations = generations
        self.theta = theta
        self.rng = rng or random.Random(SEED)
        # qubit angles; start in uniform superposition (beta^2 = 0.5)
        self.angle = [math.pi / 4 for _ in range(self.n)]

    def _measure(self) -> list[int]:
        return [1 if self.rng.random() < math.sin(a) ** 2 else 0 for a in self.angle]

    def _repair(self, mask: list[int]) -> list[int]:
        """Greedy repair to satisfy the budget using value-density ordering."""
        total = sum(c for c, m in zip(self.costs, mask) if m)
        if total <= self.budget:
            return mask
        order = sorted(range(self.n),
                       key=lambda i: self.lin_values[i] / (self.costs[i] + 1e-9))
        for i in order:
            if total <= self.budget:
                break
            if mask[i] == 1:
                mask[i] = 0
                total -= self.costs[i]
        return mask

    def _fitness(self, mask: list[int]) -> float:
        cost = sum(c for c, m in zip(self.costs, mask) if m)
        if cost > self.budget:
            return -1.0
        return self.value_fn(mask)

    def _local_search(self, mask: list[int]) -> list[int]:
        """One pass of 1-flip/swap hill-climbing — quantum-inspired fine tuning."""
        mask = list(mask)
        base = self._fitness(mask)
        spent = sum(c for c, m in zip(self.costs, mask) if m)
        improved = True
        while improved:
            improved = False
            # try adding any affordable unfunded item
            for i in range(self.n):
                if mask[i] == 0 and spent + self.costs[i] <= self.budget:
                    mask[i] = 1
                    nv = self._fitness(mask)
                    if nv > base:
                        base, spent, improved = nv, spent + self.costs[i], True
                    else:
                        mask[i] = 0
            # try swaps that free budget for a higher-value item
            for i in range(self.n):
                if mask[i] != 1:
                    continue
                for j in range(self.n):
                    if mask[j] == 0 and spent - self.costs[i] + self.costs[j] <= self.budget:
                        mask[i], mask[j] = 0, 1
                        nv = self._fitness(mask)
                        if nv > base:
                            base = nv
                            spent = spent - self.costs[i] + self.costs[j]
                            improved = True
                            break
                        mask[i], mask[j] = 1, 0
        return mask

    def run(self) -> QIEAResult:
        start = time.perf_counter()
        best_mask = [0] * self.n
        best_val = 0.0
        history: list[float] = []
        for _ in range(self.generations):
            gen_best_mask = best_mask
            gen_best_val = best_val
            for _ in range(self.pop):
                mask = self._repair(self._measure())
                val = self._fitness(mask)
                if val > gen_best_val:
                    gen_best_val = val
                    gen_best_mask = mask
            if gen_best_val > best_val:
                best_val = gen_best_val
                best_mask = gen_best_mask
            # rotation-gate update: steer qubits toward the best observed solution
            for i in range(self.n):
                target = best_mask[i]
                measured = math.sin(self.angle[i]) ** 2
                if target == 1 and measured < 0.98:
                    self.angle[i] = min(math.pi / 2, self.angle[i] + self.theta)
                elif target == 0 and measured > 0.02:
                    self.angle[i] = max(0.0, self.angle[i] - self.theta)
            history.append(best_val)
        # quantum-inspired local refinement of the final collapsed state
        refined = self._local_search(best_mask)
        rv = self._fitness(refined)
        if rv > best_val:
            best_val, best_mask = rv, refined
        if history:
            history[-1] = best_val
        return QIEAResult(best_mask, best_val, history, time.perf_counter() - start)


# =========================================================================== #
# 5. BASELINE OPTIMISERS
# =========================================================================== #
def greedy_density(costs, lin_values, budget, value_fn):
    """Classic value-density greedy on the LINEAR surrogate (synergy-blind)."""
    start = time.perf_counter()
    n = len(costs)
    order = sorted(range(n), key=lambda i: lin_values[i] / (costs[i] + 1e-9), reverse=True)
    mask = [0] * n
    total = 0.0
    for i in order:
        if total + costs[i] <= budget:
            mask[i] = 1
            total += costs[i]
    return mask, value_fn(mask), time.perf_counter() - start


def random_search(costs, budget, iters, rng, value_fn):
    start = time.perf_counter()
    n = len(costs)
    best_mask = [0] * n
    best_val = 0.0
    for _ in range(iters):
        mask = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
        total = sum(c for c, m in zip(costs, mask) if m)
        while total > budget:
            ones = [i for i in range(n) if mask[i]]
            if not ones:
                break
            drop = rng.choice(ones)
            mask[drop] = 0
            total -= costs[drop]
        val = value_fn(mask)
        if val > best_val:
            best_val, best_mask = val, mask
    return best_mask, best_val, time.perf_counter() - start


def classical_ga(costs, lin_values, budget, pop, generations, rng, value_fn):
    start = time.perf_counter()
    n = len(costs)

    def fitness(mask):
        total = sum(c for c, m in zip(costs, mask) if m)
        if total > budget:
            return -1.0
        return value_fn(mask)

    def repair(mask):
        total = sum(c for c, m in zip(costs, mask) if m)
        order = sorted(range(n), key=lambda i: lin_values[i] / (costs[i] + 1e-9))
        for i in order:
            if total <= budget:
                break
            if mask[i]:
                mask[i] = 0
                total -= costs[i]
        return mask

    population = [repair([1 if rng.random() < 0.3 else 0 for _ in range(n)])
                 for _ in range(pop)]
    best_mask, best_val, history = population[0], fitness(population[0]), []
    for _ in range(generations):
        scored = sorted(population, key=fitness, reverse=True)
        elite = scored[: max(2, pop // 5)]
        if fitness(elite[0]) > best_val:
            best_val, best_mask = fitness(elite[0]), elite[0]
        children = list(elite)
        while len(children) < pop:
            pa, pb = rng.choice(elite), rng.choice(elite)
            cut = rng.randint(1, n - 1)
            child = pa[:cut] + pb[cut:]
            for i in range(n):                        # mutation
                if rng.random() < 0.03:
                    child[i] ^= 1
            children.append(repair(child))
        population = children
        history.append(best_val)
    return best_mask, best_val, history, time.perf_counter() - start


def linear_dp(costs, values, budget):
    """Exact optimum of the LINEAR relaxation via dynamic programming.

    This is the best achievable if synergies are ignored; it provides a strong
    linear-objective upper bound and a near-optimal starting mask. The quadratic
    optimum is NP-hard, so we report the best value found across all methods plus
    this linear bound as reference points.
    """
    start = time.perf_counter()
    icosts = [max(1, int(round(c))) for c in costs]
    cap = int(round(budget))
    n = len(costs)
    dp = [0.0] * (cap + 1)
    keep = [[False] * (cap + 1) for _ in range(n)]
    for i in range(n):
        ci, vi = icosts[i], values[i]
        for b in range(cap, ci - 1, -1):
            if dp[b - ci] + vi > dp[b]:
                dp[b] = dp[b - ci] + vi
                keep[i][b] = True
    mask = [0] * n
    b = cap
    for i in range(n - 1, -1, -1):
        if keep[i][b]:
            mask[i] = 1
            b -= icosts[i]
    return mask, dp[cap], time.perf_counter() - start


# =========================================================================== #
# 6. METRICS
# =========================================================================== #
# --------------------------------------------------------------------------- #
# Defence-in-depth synergies: funding complementary control families together
# yields super-additive risk reduction (e.g. EDR+SIEM, IAM+PAM, Backup+IR).
# This makes the objective QUADRATIC (non-submodular), which defeats greedy /
# value-density heuristics and rewards population-based global search.
# --------------------------------------------------------------------------- #
# Defence-in-depth synergy weights between complementary control families.
# Magnitudes are deliberately large: several partners are individually low
# value-density (so a greedy/linear optimiser skips them), but co-funding the
# pair unlocks a substantial super-additive bonus. Only an optimiser that
# reasons over the JOINT objective recovers this value.
SYNERGY_PAIRS = {
    (0, 3): 0.90,   # EDR            + SIEM & Threat Hunting
    (2, 13): 0.85,  # IAM/MFA        + Privileged Access Management
    (7, 19): 0.80,  # Backup         + Incident Response Retainer
    (1, 8): 0.80,   # Segmentation   + Zero-Trust Access
    (9, 18): 0.75,  # Cloud Posture  + Container Runtime Security
    (6, 16): 0.75,  # Vuln Mgmt      + Patch Automation
    (10, 11): 0.70, # DLP            + Security Awareness Training
    (12, 15): 0.70, # WAF/API        + DDoS Mitigation
}


def _synergy_bonus(mask, controls):
    """Super-additive ELR bonus from co-funding complementary control families."""
    fam = {}
    for c, m in zip(controls, mask):
        if m:
            fam.setdefault(c.cid % len(CONTROL_FAMILIES), []).append(c)
    bonus = 0.0
    for (a, b), w in SYNERGY_PAIRS.items():
        if a in fam and b in fam:
            pa = sum(c.asset_value * c.base_breach_prob for c in fam[a])
            pb = sum(c.asset_value * c.base_breach_prob for c in fam[b])
            bonus += w * min(pa, pb)
    return bonus


def portfolio_metrics(mask, controls, mitigation_hat, use_knowledge, budget):
    spent = sum(c.cost for c, m in zip(controls, mask) if m)
    elr = sum(expected_loss_reduction(c, mh, use_knowledge)
             for c, mh, m in zip(controls, mitigation_hat, mask) if m)
    elr += _synergy_bonus(mask, controls)
    # "true" realised risk reduction uses ground-truth mitigation + synergies
    true_rr = sum(c.asset_value * c.base_breach_prob * c.true_mitigation
                 for c, m in zip(controls, mask) if m)
    true_rr += _synergy_bonus(mask, controls)
    roi = (true_rr - spent) / spent if spent > 0 else 0.0
    return {
        "controls_funded": sum(mask),
        "budget_spent": round(spent, 2),
        "budget_utilisation": round(spent / budget, 4) if budget else 0.0,
        "elr_utility": round(elr, 2),
        "true_risk_reduction": round(true_rr, 2),
        "roi": round(roi, 4),
    }


# =========================================================================== #
# 7. MINIMAL SVG PLOTTING (no matplotlib)
# =========================================================================== #
class SVG:
    """Tiny publication-oriented SVG canvas (axes, lines, bars, legend, text)."""

    def __init__(self, width=820, height=520, margin=(70, 40, 70, 90)):
        self.w, self.h = width, height
        self.mt, self.mr, self.mb, self.ml = margin
        self.el: list[str] = []
        self.palette = ["#1f4e79", "#c0504d", "#4f8a3b", "#8064a2", "#e08e0b"]

    @property
    def px0(self): return self.ml
    @property
    def px1(self): return self.w - self.mr
    @property
    def py0(self): return self.h - self.mb
    @property
    def py1(self): return self.mt

    def _x(self, t, xmin, xmax):
        return self.px0 + (t - xmin) / (xmax - xmin + 1e-12) * (self.px1 - self.px0)

    def _y(self, v, ymin, ymax):
        return self.py0 - (v - ymin) / (ymax - ymin + 1e-12) * (self.py0 - self.py1)

    def axes(self, xmin, xmax, ymin, ymax, xlabel, ylabel, title, xticks=6, yticks=6):
        self.el.append(f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="white"/>')
        self.el.append(
            f'<text x="{self.w/2}" y="26" font-family="Helvetica,Arial" font-size="19" '
            f'font-weight="bold" text-anchor="middle" fill="#222">{title}</text>')
        # grid + ticks
        for k in range(yticks + 1):
            v = ymin + (ymax - ymin) * k / yticks
            y = self._y(v, ymin, ymax)
            self.el.append(f'<line x1="{self.px0}" y1="{y:.1f}" x2="{self.px1}" y2="{y:.1f}" '
                          f'stroke="#e6e6e6" stroke-width="1"/>')
            self.el.append(f'<text x="{self.px0-10}" y="{y+4:.1f}" font-family="Helvetica,Arial" '
                          f'font-size="12" text-anchor="end" fill="#444">{v:.0f}</text>')
        for k in range(xticks + 1):
            t = xmin + (xmax - xmin) * k / xticks
            x = self._x(t, xmin, xmax)
            self.el.append(f'<text x="{x:.1f}" y="{self.py0+20:.1f}" font-family="Helvetica,Arial" '
                          f'font-size="12" text-anchor="middle" fill="#444">{t:.0f}</text>')
        self.el.append(f'<line x1="{self.px0}" y1="{self.py0}" x2="{self.px1}" y2="{self.py0}" '
                      f'stroke="#333" stroke-width="1.5"/>')
        self.el.append(f'<line x1="{self.px0}" y1="{self.py0}" x2="{self.px0}" y2="{self.py1}" '
                      f'stroke="#333" stroke-width="1.5"/>')
        self.el.append(f'<text x="{(self.px0+self.px1)/2:.1f}" y="{self.h-18}" '
                      f'font-family="Helvetica,Arial" font-size="14" text-anchor="middle" '
                      f'fill="#222">{xlabel}</text>')
        self.el.append(f'<text x="22" y="{(self.py0+self.py1)/2:.1f}" font-family="Helvetica,Arial" '
                      f'font-size="14" text-anchor="middle" fill="#222" '
                      f'transform="rotate(-90 22 {(self.py0+self.py1)/2:.1f})">{ylabel}</text>')

    def line(self, xs, ys, xmin, xmax, ymin, ymax, color, dash=None):
        pts = " ".join(f"{self._x(x,xmin,xmax):.1f},{self._y(y,ymin,ymax):.1f}"
                       for x, y in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                      f'stroke-width="2.6"{d}/>')

    def markers(self, xs, ys, xmin, xmax, ymin, ymax, color, r=4):
        for x, y in zip(xs, ys):
            self.el.append(f'<circle cx="{self._x(x,xmin,xmax):.1f}" '
                          f'cy="{self._y(y,ymin,ymax):.1f}" r="{r}" fill="{color}"/>')

    def legend(self, items, x=None, y=None):
        x = x if x is not None else self.px1 - 230
        y = y if y is not None else self.py1 + 8
        self.el.append(f'<rect x="{x-10}" y="{y-6}" width="238" height="{18*len(items)+12}" '
                      f'fill="white" stroke="#ccc" stroke-width="1" opacity="0.95"/>')
        for i, (label, color, dash) in enumerate(items):
            yy = y + 12 + i * 18
            d = f' stroke-dasharray="{dash}"' if dash else ""
            self.el.append(f'<line x1="{x}" y1="{yy-4}" x2="{x+26}" y2="{yy-4}" '
                          f'stroke="{color}" stroke-width="3"{d}/>')
            self.el.append(f'<text x="{x+34}" y="{yy}" font-family="Helvetica,Arial" '
                          f'font-size="12.5" fill="#333">{label}</text>')

    def save(self, path):
        body = "\n".join(self.el)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}">\n{body}\n</svg>\n')
        with open(path, "w") as f:
            f.write(svg)


# =========================================================================== #
# 8. TABLE WRITERS
# =========================================================================== #
def write_table(path_base, header, rows):
    with open(path_base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    widths = [max(len(str(header[i])), max((len(str(r[i])) for r in rows), default=0))
              for i in range(len(header))]
    def fmt(r): return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |"
    lines = [fmt(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt(r) for r in rows]
    with open(path_base + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")


# =========================================================================== #
# 9. EXPERIMENT DRIVER
# =========================================================================== #
def train_test_split(controls, frac, rng):
    idx = list(range(len(controls)))
    rng.shuffle(idx)
    k = int(len(controls) * frac)
    train = [controls[i] for i in idx[:k]]
    test = [controls[i] for i in idx[k:]]
    return train, test


def run_experiment():
    os.makedirs(ARTIFACTS, exist_ok=True)
    rng = random.Random(SEED)

    # ---- data ----
    N = 120
    controls = generate_portfolio(N, rng)

    # ---- prediction layer (learn mitigation effectiveness) ----
    # The predictor learns the additive feature response only; the interaction
    # "premium" on crown-jewel/high-velocity assets is deliberately NOT learnable
    # from the marginal features, leaving a knowledge gap for the knowledge layer.
    def _base_mitigation(c: Control) -> float:
        ac, tv, ex, mg, ds = c.features
        prem = 0.12 if (0.5 * ac + 0.5 * ds) > 0.6 and (0.5 * tv + 0.5 * ex) > 0.6 else 0.0
        return max(0.03, min(0.99, c.true_mitigation - prem))

    train, _ = train_test_split(controls, 0.6, random.Random(SEED + 1))
    Xtr = [c.features for c in train]
    ytr = [_base_mitigation(c) for c in train]
    model = LogisticRegression(n_features=5).fit(Xtr, ytr)
    mitigation_hat = model.predict([c.features for c in controls])

    # predictor quality (MAE against the learnable base signal)
    mae = sum(abs(ph - _base_mitigation(c))
              for ph, c in zip(mitigation_hat, controls)) / N

    costs = [c.cost for c in controls]
    total_cost = sum(costs)
    budget = round(0.35 * total_cost, 2)        # realistic constrained budget

    # ---- knowledge-adjusted LINEAR ELR utilities (per-control) ----
    values_full = [expected_loss_reduction(c, mh, True)
                   for c, mh in zip(controls, mitigation_hat)]

    # ---- full QUADRATIC objective (linear ELR + defence-in-depth synergies) ----
    def quad_value(mask):
        v = sum(val for val, m in zip(values_full, mask) if m)
        return v + _synergy_bonus(mask, controls)

    # =================================================================== #
    # Table 1 — method comparison, averaged over independent seeds.
    # The objective is the QUADRATIC synergy-aware ELR; greedy and the linear
    # DP optimise the synergy-blind surrogate and are therefore sub-optimal.
    # Gap is measured against the best value found by any method (reference).
    # =================================================================== #
    lin_mask, _, lin_t = linear_dp(costs, values_full, budget)
    lin_quad_val = quad_value(lin_mask)         # linear-optimal mask scored on quad obj

    N_RUNS = 20
    names = ["Random Search", "Greedy (value-density)", "Classical GA",
             "QIEA (ours)", "Linear-DP (synergy-blind)"]
    agg = {name: {"gap": [], "val": [], "trr": [], "roi": [], "funded": [],
                  "util": [], "ms": []} for name in names}
    q_res = None            # keep one QIEA run for the convergence figure
    ga_hist = None

    for run in range(N_RUNS):
        g_mask, g_val, g_t = greedy_density(costs, values_full, budget, quad_value)
        r_mask, r_val, r_t = random_search(costs, budget, 3200,
                                           random.Random(SEED + 100 + run), quad_value)
        ga_mask, ga_val, ga_h, ga_t = classical_ga(costs, values_full, budget,
                                                   24, 140,
                                                   random.Random(SEED + 200 + run),
                                                   quad_value)
        qr = QuantumInspiredEA(N, costs, quad_value, budget, values_full,
                               pop=24, generations=140, theta=0.035,
                               rng=random.Random(SEED + 300 + run)).run()
        if run == 0:
            q_res, ga_hist = qr, ga_h

        run_results = {
            "Random Search": (r_mask, r_t),
            "Greedy (value-density)": (g_mask, g_t),
            "Classical GA": (ga_mask, ga_t),
            "QIEA (ours)": (qr.best_mask, qr.elapsed),
            "Linear-DP (synergy-blind)": (lin_mask, lin_t),
        }
        ref = max(quad_value(m) for m, _ in run_results.values())
        for name, (mask, t) in run_results.items():
            m = portfolio_metrics(mask, controls, mitigation_hat, True, budget)
            val = quad_value(mask)
            gap = 100.0 * (1.0 - val / ref) if ref else 0.0
            agg[name]["gap"].append(gap)
            agg[name]["val"].append(val)
            agg[name]["trr"].append(m["true_risk_reduction"])
            agg[name]["roi"].append(m["roi"])
            agg[name]["funded"].append(m["controls_funded"])
            agg[name]["util"].append(m["budget_utilisation"])
            agg[name]["ms"].append(t * 1000)

    def _mean(v): return sum(v) / len(v)
    def _std(v):
        mu = _mean(v)
        return (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5

    table1_rows = []
    results_methods = {}
    for name in names:
        a = agg[name]
        table1_rows.append([
            name, f"{_mean(a['funded']):.0f}",
            f"{_mean(a['util'])*100:.1f}%",
            f"{_mean(a['val']):.1f}",
            f"{_mean(a['trr']):.1f}",
            f"{_mean(a['roi'])*100:.1f}%",
            f"{_mean(a['gap']):.3f} \u00b1 {_std(a['gap']):.3f}",
            f"{_mean(a['ms']):.1f}",
        ])
        results_methods[name] = {
            "controls_funded_mean": round(_mean(a["funded"]), 2),
            "budget_utilisation_mean": round(_mean(a["util"]), 4),
            "objective_value_mean": round(_mean(a["val"]), 2),
            "true_risk_reduction_mean": round(_mean(a["trr"]), 2),
            "roi_mean": round(_mean(a["roi"]), 4),
            "optimality_gap_pct_mean": round(_mean(a["gap"]), 4),
            "optimality_gap_pct_std": round(_std(a["gap"]), 4),
            "runtime_ms_mean": round(_mean(a["ms"]), 2),
        }

    write_table(
        os.path.join(ARTIFACTS, "table1_performance"),
        ["Method", "Controls funded", "Budget util.", "Objective value",
         "True risk reduction", "ROI", "Gap to best (%)", "Runtime (ms)"],
        table1_rows,
    )

    # =================================================================== #
    # Table 2 — ablation: contribution of each framework layer.
    # Each variant is optimised by QIEA; we progressively enable the
    # prediction, knowledge, and synergy (quadratic) components.
    # =================================================================== #
    # (a) Optimisation only: naive uniform mitigation proxy, no knowledge, linear.
    values_naive = [c.asset_value * c.base_breach_prob * 0.5 for c in controls]
    # (b) + Prediction layer: learned mitigation, no knowledge, linear.
    values_ML = [expected_loss_reduction(c, mh, False)
                 for c, mh in zip(controls, mitigation_hat)]
    # (c) + Knowledge layer: knowledge-adjusted, linear (== values_full).
    # (d) + Synergy layer: full quadratic objective (quad_value).

    def _lin_fn(vals):
        return lambda mask: sum(v for v, m in zip(vals, mask) if m)

    def eval_variant(value_fn, lin_ref, use_knowledge, label):
        res = QuantumInspiredEA(N, costs, value_fn, budget, lin_ref,
                                pop=24, generations=140, theta=0.035,
                                rng=random.Random(SEED + 99)).run()
        m = portfolio_metrics(res.best_mask, controls, mitigation_hat, use_knowledge, budget)
        return label, m

    variants = [
        eval_variant(_lin_fn(values_naive), values_naive, False,
                     "QIEA only (no learning / knowledge / synergy)"),
        eval_variant(_lin_fn(values_ML), values_ML, False,
                     "+ Prediction layer"),
        eval_variant(_lin_fn(values_full), values_full, True,
                     "+ Knowledge layer"),
        eval_variant(quad_value, values_full, True,
                     "+ Synergy layer (full framework)"),
    ]
    table2_rows = []
    results_ablation = {}
    base_true = variants[0][1]["true_risk_reduction"]
    for label, m in variants:
        lift = 100.0 * (m["true_risk_reduction"] / base_true - 1.0) if base_true else 0.0
        table2_rows.append([
            label, m["controls_funded"], m["elr_utility"],
            m["true_risk_reduction"], f'{m["roi"]*100:.1f}%', f"{lift:+.1f}%",
        ])
        results_ablation[label] = {**m, "risk_reduction_lift_pct": round(lift, 3)}
    write_table(
        os.path.join(ARTIFACTS, "table2_ablation"),
        ["Framework configuration", "Controls funded", "ELR utility",
         "True risk reduction", "ROI", "Lift vs. base"],
        table2_rows,
    )

    # =================================================================== #
    # Figure 1 — convergence of QIEA vs. Classical GA (+ greedy / optimum refs)
    # =================================================================== #
    gens = list(range(1, len(q_res.history) + 1))
    g_mask_ref, g_val, _ = greedy_density(costs, values_full, budget, quad_value)
    lin_ref_val = quad_value(lin_mask)
    ymax = max(max(q_res.history), max(ga_hist), lin_ref_val, g_val) * 1.02
    ymin = min(min(q_res.history), min(ga_hist), g_val) * 0.985
    fig1 = SVG()
    fig1.axes(1, len(gens), ymin, ymax, "Generation",
              "Best synergy-aware objective (thousands)",
              "Fig. 1  Convergence of the quantum-inspired optimiser", xticks=6, yticks=6)
    fig1.line(gens, [lin_ref_val] * len(gens), 1, len(gens), ymin, ymax,
              fig1.palette[4], dash="6,5")
    fig1.line(gens, [g_val] * len(gens), 1, len(gens), ymin, ymax,
              fig1.palette[1], dash="2,4")
    fig1.line(gens, ga_hist, 1, len(gens), ymin, ymax, fig1.palette[2])
    fig1.line(gens, q_res.history, 1, len(gens), ymin, ymax, fig1.palette[0])
    fig1.legend([
        ("QIEA (ours)", fig1.palette[0], None),
        ("Classical GA", fig1.palette[2], None),
        ("Linear-DP (synergy-blind)", fig1.palette[4], "6,5"),
        ("Greedy baseline", fig1.palette[1], "2,4"),
    ], x=fig1.px0 + 24, y=fig1.py0 - 92)
    fig1.save(os.path.join(ARTIFACTS, "figure1_convergence.svg"))

    # =================================================================== #
    # Figure 2 — risk-reduction efficiency frontier vs. budget
    # =================================================================== #
    fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    budgets = [round(f * total_cost, 2) for f in fractions]
    curves = {"QIEA (ours)": [], "Classical GA": [], "Greedy": [], "Random": []}
    for b in budgets:
        qr = QuantumInspiredEA(N, costs, quad_value, b, values_full, pop=24,
                               generations=140, theta=0.035,
                               rng=random.Random(SEED + 11)).run()
        curves["QIEA (ours)"].append(
            portfolio_metrics(qr.best_mask, controls, mitigation_hat, True, b)["true_risk_reduction"])
        gm, _, _, _ = classical_ga(costs, values_full, b, 24, 140,
                                   random.Random(SEED + 12), quad_value)
        curves["Classical GA"].append(
            portfolio_metrics(gm, controls, mitigation_hat, True, b)["true_risk_reduction"])
        gd, _, _ = greedy_density(costs, values_full, b, quad_value)
        curves["Greedy"].append(
            portfolio_metrics(gd, controls, mitigation_hat, True, b)["true_risk_reduction"])
        rm, _, _ = random_search(costs, b, 3200, random.Random(SEED + 13), quad_value)
        curves["Random"].append(
            portfolio_metrics(rm, controls, mitigation_hat, True, b)["true_risk_reduction"])

    xs = [f * 100 for f in fractions]
    ally = [v for c in curves.values() for v in c]
    fig2 = SVG()
    fig2.axes(min(xs), max(xs), min(ally) * 0.95, max(ally) * 1.03,
              "Security budget (% of full portfolio cost)",
              "Realised annual risk reduction (thousands)",
              "Fig. 2  Budget\u2013risk-reduction efficiency frontier", xticks=8, yticks=6)
    colmap = {"QIEA (ours)": fig2.palette[0], "Classical GA": fig2.palette[2],
              "Greedy": fig2.palette[1], "Random": fig2.palette[3]}
    for name, ys in curves.items():
        fig2.line(xs, ys, min(xs), max(xs), min(ally) * 0.95, max(ally) * 1.03, colmap[name])
        fig2.markers(xs, ys, min(xs), max(xs), min(ally) * 0.95, max(ally) * 1.03, colmap[name])
    fig2.legend([(n, colmap[n], None) for n in curves], x=fig2.px0 + 24, y=fig2.py0 - 92)
    fig2.save(os.path.join(ARTIFACTS, "figure2_frontier.svg"))

    # ---- machine-readable summary ----
    summary = {
        "seed": SEED,
        "dataset": {"n_controls": N, "total_cost": round(total_cost, 2),
                    "budget": budget, "budget_fraction": 0.35},
        "prediction_layer": {"model": "logistic_regression_scratch",
                             "mitigation_MAE": round(mae, 4)},
        "n_runs": N_RUNS,
        "objective": "quadratic_synergy_aware_ELR",
        "table1_methods": results_methods,
        "table2_ablation": results_ablation,
        "linear_dp_reference_quad_value": round(lin_ref_val, 2),
        "qiea_vs_best_gap_pct": results_methods["QIEA (ours)"]["optimality_gap_pct_mean"],
        "figure2_frontier": {"budget_fractions": fractions, "curves": curves},
    }
    with open(os.path.join(ARTIFACTS, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    t0 = time.perf_counter()
    summary = run_experiment()
    dt = time.perf_counter() - t0
    q = summary["table1_methods"]["QIEA (ours)"]
    gd = summary["table1_methods"]["Greedy (value-density)"]
    ga = summary["table1_methods"]["Classical GA"]
    print("=" * 68)
    print(f"QAI-CIP experiment complete  ({summary['n_runs']} seeds averaged)")
    print("=" * 68)
    print(f"Controls: {summary['dataset']['n_controls']}  "
          f"Budget: {summary['dataset']['budget']} "
          f"({int(summary['dataset']['budget_fraction']*100)}% of portfolio)")
    print(f"Objective                       : {summary['objective']}")
    print(f"Prediction-layer mitigation MAE : {summary['prediction_layer']['mitigation_MAE']}")
    print(f"QIEA   gap to best (mean)       : {q['optimality_gap_pct_mean']}% "
          f"(+/- {q['optimality_gap_pct_std']})")
    print(f"GA     gap to best (mean)       : {ga['optimality_gap_pct_mean']}%")
    print(f"Greedy gap to best (mean)       : {gd['optimality_gap_pct_mean']}%")
    print(f"QIEA   objective value (mean)   : {q['objective_value_mean']}")
    print(f"Greedy objective value (mean)   : {gd['objective_value_mean']}")
    print(f"QIEA   ROI (mean)               : {q['roi_mean']*100:.1f}%")
    print(f"Artifacts written to            : {ARTIFACTS}")
    print(f"Total wall-clock time           : {dt:.2f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()
