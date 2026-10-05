"""Statistics for comparing optimizers: bootstrap CIs, Mann-Whitney U, success rates.

Runs that never reach the spec are right-censored: they are scored as
`budget + 1` simulations, so every statistic stays defined and conservative.
"""
from __future__ import annotations

import math

import numpy as np


def sims_to_spec(first_feasible: int | None, budget: int) -> int:
    return first_feasible if first_feasible is not None else budget + 1


def bootstrap_ci(x, stat=np.median, n: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    s = np.array([stat(rng.choice(x, len(x))) for _ in range(n)])
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success rate k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mann_whitney_p(a, b) -> float:
    """Two-sided Mann-Whitney U p-value (normal approximation, tie-corrected)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(allv, return_counts=True)
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - (counts ** 3 - counts).sum() / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (abs(u - n1 * n2 / 2) - 0.5) / math.sqrt(var)
    return math.erfc(max(z, 0.0) / math.sqrt(2))


def summarize(runs: list[dict], budget: int) -> dict:
    """Per-algorithm summary with CIs from benchmark records."""
    out = {}
    for algo in sorted({r["algo"] for r in runs}):
        rs = [r for r in runs if r["algo"] == algo]
        s = [sims_to_spec(r["first_feasible"], budget) for r in rs]
        ok = sum(r["first_feasible"] is not None for r in rs)
        out[algo] = {"n": len(rs), "success": ok, "success_ci": wilson_ci(ok, len(rs)),
                     "median_sims": float(np.median(s)), "median_sims_ci": bootstrap_ci(s), "sims": s}
    return out


def pairwise_pvalues(summary: dict) -> dict:
    algos = list(summary)
    return {f"{a} vs {b}": mann_whitney_p(summary[a]["sims"], summary[b]["sims"])
            for i, a in enumerate(algos) for b in algos[i + 1:]}
