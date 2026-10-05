"""PyTorch surrogate of SPICE performance and a surrogate-assisted GA.

The surrogate is a small ensemble of MLPs mapping a sizing vector to op-amp
metrics. Ensemble spread gives a cheap uncertainty estimate used to pick
which candidates are worth a real SPICE run.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist

import numpy as np
import torch

torch.set_num_threads(1)
from torch import nn

from .circuit import GROUPS, MULT_KEYS, Design
from .optimizers import cardinality, decode, mutate_gene, random_vector, run_safely
from .specs import Evaluator, Spec, violations

TARGETS = ["gain_db", "log_ugbw", "pm_deg", "log_power", "sat_margin_v", "vout_err_v"]


def features(d: Design) -> list[float]:
    f = []
    for g in GROUPS:
        w, l = d.wl(g)
        f += [math.log(w), math.log(l)]
    f += [math.log(d.mult[k]) for k in MULT_KEYS]
    f += [math.log(d.cc), d.rz / 10.0, math.log(d.ib)]
    return f


def targets(m: dict) -> list[float]:
    return [max(m["gain_db"], -20.0), math.log10(max(m["ugbw_mhz"], 1e-3)), max(m["pm_deg"], -90.0),
            math.log10(max(m["power_uw"], 1e-3)), max(m["min_sat_margin_mv"], -1000.0) / 1e3,
            abs(m["vout_dc"] - 0.9)]


def to_metrics(y: np.ndarray) -> dict:
    return {"gain_db": y[0], "ugbw_mhz": 10 ** y[1], "pm_deg": y[2], "power_uw": 10 ** y[3],
            "min_sat_margin_mv": y[4] * 1e3, "vout_dc": 0.9 + y[5]}


class Surrogate:
    def __init__(self, n_models: int = 4, hidden: int = 128, epochs: int = 300, seed: int = 0):
        self.n_models, self.hidden, self.epochs, self.seed = n_models, hidden, epochs, seed
        self.models: list[nn.Module] = []

    def _net(self, din: int, dout: int) -> nn.Module:
        return nn.Sequential(nn.Linear(din, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.hidden),
                             nn.SiLU(), nn.Linear(self.hidden, dout))

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Surrogate":
        torch.manual_seed(self.seed)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-8
        Xt = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        Yt = torch.tensor((Y - self.ym) / self.ys, dtype=torch.float32)
        self.models = []
        n = len(Xt)
        for k in range(self.n_models):
            net = self._net(X.shape[1], Y.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
            g = torch.Generator().manual_seed(self.seed * 100 + k)
            idx = torch.randint(0, n, (n,), generator=g)          # bootstrap per member
            xb, yb = Xt[idx], Yt[idx]
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                for i in range(0, n, 128):
                    j = perm[i:i + 128]
                    loss = nn.functional.smooth_l1_loss(net(xb[j]), yb[j])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            self.models.append(net.eval())
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xt = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        P = np.stack([m(Xt).numpy() for m in self.models]) * self.ys + self.ym
        return P.mean(0), P.std(0)


def predicted_cost(y: np.ndarray, spec: Spec) -> float:
    return sum(violations(to_metrics(y), spec).values())


@run_safely
def surrogate_ga(ev: Evaluator, seed: int = 0, warmup: int = 40, per_gen: int = 10,
                 pool: int = 400, epochs: int = 150, min_train: int = 8, explore: float = 0.5,
                 calibrate: bool = False) -> None:
    """Surrogate-assisted GA: an MLP ensemble screens `pool` GA offspring per
    generation; only the `per_gen` most promising (lower-confidence-bound on
    predicted violation) are simulated in SPICE. Retrains every generation."""
    rng = random.Random(seed)
    card = cardinality()
    seen: list[tuple[list[int], float, dict | None]] = []

    def run(v):
        c, m, _ = ev(decode(v))
        seen.append((v, c, m))

    for _ in range(warmup):
        run(random_vector(rng))
    gen = 0
    while True:
        good = [(v, m) for v, _, m in seen if m is not None]
        while len(good) < min_train:         # too few successful sims to fit a model: keep sampling
            v = random_vector(rng)
            run(v)
            if seen[-1][2] is not None:
                good.append((v, seen[-1][2]))
        X = np.array([features(decode(v)) for v, _ in good])
        Y = np.array([targets(m) for _, m in good])
        sur: Surrogate | CalibratedSurrogate
        if calibrate:
            sur = CalibratedSurrogate(lambda: Surrogate(n_models=3, epochs=epochs, seed=seed + gen), seed=seed + gen).fit(X, Y)
        else:
            sur = Surrogate(n_models=3, epochs=epochs, seed=seed + gen).fit(X, Y)
        parents = sorted(seen, key=lambda t: t[1])[:12]
        cands: list[list[int]] = []
        keys = set(ev.cache)
        while len(cands) < pool:
            a, b = rng.choice(parents)[0], rng.choice(parents)[0]
            child = [x if rng.random() < 0.5 else y for x, y in zip(a, b)]
            for i in range(len(child)):
                if rng.random() < 0.2:
                    mutate_gene(child, i, rng, card)
            if decode(child).key() not in keys:
                keys.add(decode(child).key())
                cands.append(child)
        mu, sd = sur.predict(np.array([features(decode(v)) for v in cands]))
        # Optimistic score: predicted violation evaluated at mean, minus an exploration bonus.
        score = [predicted_cost(mu[i], ev.spec) - explore * float((sd[i] / sur.ys).mean()) for i in range(len(cands))]
        for i in np.argsort(score)[:per_gen]:
            run(cands[i])
        gen += 1


class CalibratedSurrogate:
    """Split-conformal rescaling of any surrogate's predicted std.

    Fits the base model on (1 - cal_frac) of the data, then per target scales its std so the
    +/- 1.645 sd interval covers `level` of the held-out calibration errors. The mean prediction is
    unchanged; only the uncertainty is corrected (ensembles are typically overconfident).
    """

    def __init__(self, base_factory, cal_frac: float = 0.2, level: float = 0.9, seed: int = 0):
        self.base_factory, self.cal_frac, self.level, self.seed = base_factory, cal_frac, level, seed

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CalibratedSurrogate":
        perm = np.random.default_rng(self.seed).permutation(len(X))
        n_cal = max(5, int(self.cal_frac * len(X)))
        cal, tr = perm[:n_cal], perm[n_cal:]
        self.base = self.base_factory().fit(X[tr], Y[tr])
        mu, sd = self.base.predict(X[cal])
        z = np.abs(Y[cal] - mu) / np.maximum(sd, 1e-9)
        # sd is inflated or shrunk so the nominal Gaussian interval hits the empirical quantile
        self.scale = np.quantile(z, self.level, axis=0) / NormalDist().inv_cdf(0.5 + self.level / 2)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = self.base.predict(X)
        return mu, sd * self.scale

    @property
    def ys(self) -> np.ndarray:
        return self.base.ys
