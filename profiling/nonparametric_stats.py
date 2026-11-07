#!/usr/bin/env python3
"""
nonparametric_stats.py — the ONE shared stats helper for regression verdicts.

Implements the two tests Liao et al. (ICSE'25) use for both their local and
system-level signals, so this module is the single consumer-shared implementation
(per the system oracle protocol §3):

  • Wilcoxon rank-sum (a.k.a. Mann-Whitney U), two-sided, normal approximation
    with tie correction — robust to the outlier-heavy JVM timing distributions
    where a mean-based 2σ guard under-detects (validated live on PERF-307-dualval:
    +181.8% mean / +87% median slowdown was held "suspect" because a few GC/JIT
    outliers inflated stddev above the mean).
  • Cliff's Delta effect size with the conventional magnitude labels
    (|d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large).

Pure stdlib — no scipy dependency at runtime. The normal approximation is the
standard large-sample form and is adequate for n >= ~8 per side; below that the
p-value is conservative rather than anti-conservative (ties handled explicitly).
"""
import math
from collections import Counter


def _rank(values: list[float]) -> list[float]:
    """Midranks (average rank for ties), 1-based, aligned with `values` order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    return ranks


def _norm_sf(z: float) -> float:
    """Standard normal survival function P(Z > z) via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_rank_sum(a: list[float], b: list[float]) -> float:
    """Two-sided Wilcoxon rank-sum p-value for samples `a` vs `b`.

    Normal approximation with continuity correction and tie correction.
    Returns 1.0 when either sample is empty or both are constant-identical
    (no evidence of difference).
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 1.0
    combined = list(a) + list(b)
    ranks = _rank(combined)
    ra = sum(ranks[:na])                      # rank sum of sample a
    mu = na * (nb + na + 1) / 2.0             # E[ra] under H0
    n = na + nb
    tie_counts = Counter(combined).values()
    tie_term = sum(t ** 3 - t for t in tie_counts)
    var = (na * nb / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return 1.0                            # all values tied — indistinguishable
    z = (abs(ra - mu) - 0.5) / math.sqrt(var)  # continuity correction
    z = max(z, 0.0)
    return min(1.0, 2.0 * _norm_sf(z))


def cliffs_delta(a: list[float], b: list[float]) -> tuple[float, str]:
    """Cliff's Delta of `b` relative to `a`: P(b > a) - P(b < a) over all pairs.

    Positive delta means `b` values tend LARGER than `a` (for exec times:
    candidate slower than baseline). Returns (delta, magnitude_label).

    O((n+m)·log n) via sort + bisect (was O(n·m) pairwise). The induced-inefficiency
    junior inflates a hot method's call count into the tens/hundreds of thousands, where
    the pairwise form is ~10^10 comparisons and effectively hangs. Result is identical:
    for each candidate x, (# baseline < x) contributes to gt and (# baseline > x) to lt.
    """
    import bisect
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0, "negligible"
    a_sorted = sorted(a)
    gt = lt = 0
    for x in b:
        gt += bisect.bisect_left(a_sorted, x)          # baseline values strictly < x
        lt += na - bisect.bisect_right(a_sorted, x)     # baseline values strictly > x
    delta = (gt - lt) / (na * nb)
    mag = abs(delta)
    if mag < 0.147:
        label = "negligible"
    elif mag < 0.33:
        label = "small"
    elif mag < 0.474:
        label = "medium"
    else:
        label = "large"
    return delta, label
