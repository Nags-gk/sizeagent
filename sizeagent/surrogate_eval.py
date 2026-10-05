"""Metrics for judging a surrogate by what it is used for: ranking designs for SPICE.

Accuracy (R^2/MAE) is necessary but not sufficient; the surrogate-assisted GA only needs the
predicted-best designs to be genuinely good, and needs sd to say when a prediction is shaky.
"""
from __future__ import annotations

import numpy as np

from .specs import Spec, violations
from .surrogate import to_metrics


def r2_mae(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss_tot), float(np.abs(y - p).mean())


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(np.asarray(a, float)), _ranks(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def calibration(y: np.ndarray, mu: np.ndarray, sd: np.ndarray, z: float = 1.645) -> dict:
    """Coverage of the +/- z*sd interval (nominal 90% for z=1.645) and how well sd ranks the errors."""
    err = np.abs(y - mu)
    return {"coverage90": float((err <= z * np.maximum(sd, 1e-12)).mean()),
            "sd_error_spearman": spearman(sd, err), "mean_sd": float(sd.mean()), "mean_abs_err": float(err.mean())}


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a random positive < score of a random negative); lower score = more likely positive."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _ranks(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(1 - u / (len(pos) * len(neg)))


def predicted_violation(Y_pred: np.ndarray, spec: Spec) -> np.ndarray:
    return np.array([sum(violations(to_metrics(y), spec).values()) for y in Y_pred])


def feasibility_ranking(Y_true: np.ndarray, Y_pred: np.ndarray, spec: Spec, ks=(10, 25, 50)) -> dict:
    """How well predicted violation ranks truly-feasible designs: AUROC and precision@k."""
    true_v = predicted_violation(Y_true, spec)
    feasible = true_v == 0
    pred_v = predicted_violation(Y_pred, spec)
    order = np.argsort(pred_v, kind="mergesort")
    out = {"n_feasible": int(feasible.sum()), "n": len(feasible), "auroc": auroc(pred_v, feasible)}
    for k in ks:
        k = min(k, len(order))
        out[f"precision_at_{k}"] = float(feasible[order[:k]].mean())
    out["cost_spearman"] = spearman(pred_v, true_v)
    return out
