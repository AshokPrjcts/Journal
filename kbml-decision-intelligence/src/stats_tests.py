"""
stats_tests.py — Lightweight statistical tests (pure Python standard library).

Provides a paired comparison with a two-sided p-value for matched per-split
results, mirroring the paired significance testing used in high-quality
Knowledge-Based Systems short communications. We use a paired t-test (exact
mean/variance) and report Cohen's d_z effect size; a normal approximation is
used for the p-value, which is accurate for the sample sizes reported here.
"""

from __future__ import annotations
import math
from typing import List, Tuple


def _erf(x: float) -> float:
    # Abramowitz & Stegun 7.1.26 approximation of the error function.
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
               - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return math.copysign(y, x)


def _norm_sf(z: float) -> float:
    """Two-sided survival probability for a standard normal |Z| >= |z|."""
    return 2.0 * (0.5 * (1.0 - _erf(abs(z) / math.sqrt(2.0))))


def paired_test(a: List[float], b: List[float]) -> Tuple[float, float, float]:
    """Paired comparison of two matched samples (a vs b).

    Returns (mean_difference, cohens_dz, p_value_two_sided).
    Positive mean_difference means a > b on average.
    """
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    if n < 2:
        return md, 0.0, 1.0
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 1e-12
    dz = md / sd                       # Cohen's d_z (paired effect size)
    t = md / (sd / math.sqrt(n))       # paired t statistic
    p = _norm_sf(t)                    # normal approximation to the p-value
    return md, dz, p


def fmt_p(p: float) -> str:
    """Format a p-value compactly (e.g., 3.2e-04 or 0.013)."""
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"
