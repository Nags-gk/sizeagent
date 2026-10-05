"""Alternative surrogates with the same interface as `sizeagent.surrogate.Surrogate`:
`fit(X, Y) -> self` and `predict(X) -> (mean, std)`, both shaped (n, n_targets).

Requires scikit-learn (`pip install -e .[research]`).
"""
from __future__ import annotations

import warnings

import numpy as np


class _Standardized:
    """Shared input/output standardization."""

    def _prep(self, X: np.ndarray, Y: np.ndarray | None = None):
        if Y is not None:
            self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
            self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-8
        Xn = (X - self.xm) / self.xs
        return Xn if Y is None else (Xn, (Y - self.ym) / self.ys)


class GPSurrogate(_Standardized):
    """Gaussian process (Matern 5/2 + noise, one ARD length-scale per input, shared across targets)."""

    def __init__(self, seed: int = 0, restarts: int = 1):
        self.seed, self.restarts = seed, restarts

    def fit(self, X: np.ndarray, Y: np.ndarray) -> GPSurrogate:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        Xn, Yn = self._prep(X, Y)
        k = (ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=np.ones(X.shape[1]) * 3.0,
                                                        length_scale_bounds=(1e-2, 1e3), nu=2.5)
             + WhiteKernel(1e-2, (1e-6, 1.0)))
        self.gp = GaussianProcessRegressor(k, n_restarts_optimizer=self.restarts, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(Xn, Yn)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = self.gp.predict(self._prep(X), return_std=True)
        mu = mu.reshape(len(X), -1)
        sd = np.asarray(sd)
        if sd.ndim == 1:                       # older scikit-learn: one std shared by all targets
            sd = np.repeat(sd.reshape(-1, 1), mu.shape[1], 1)
        return mu * self.ys + self.ym, sd * self.ys


class ForestSurrogate(_Standardized):
    """Random forest; the spread across trees is the uncertainty estimate."""

    def __init__(self, n_trees: int = 300, seed: int = 0):
        self.n_trees, self.seed = n_trees, seed

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ForestSurrogate:
        from sklearn.ensemble import RandomForestRegressor
        Xn, Yn = self._prep(X, Y)
        self.rf = RandomForestRegressor(self.n_trees, min_samples_leaf=2, n_jobs=1, random_state=self.seed).fit(Xn, Yn)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xn = self._prep(X)
        P = np.stack([t.predict(Xn) for t in self.rf.estimators_]).reshape(len(self.rf.estimators_), len(X), -1)
        return P.mean(0) * self.ys + self.ym, P.std(0) * self.ys


class BoostSurrogate(_Standardized):
    """Bagged histogram gradient boosting (one model per target and bootstrap member)."""

    def __init__(self, n_bags: int = 5, seed: int = 0):
        self.n_bags, self.seed = n_bags, seed

    def fit(self, X: np.ndarray, Y: np.ndarray) -> BoostSurrogate:
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xn, Yn = self._prep(X, Y)
        rng = np.random.default_rng(self.seed)
        self.bags = []
        for b in range(self.n_bags):
            idx = rng.integers(0, len(Xn), len(Xn))
            self.bags.append([HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=b)
                              .fit(Xn[idx], Yn[idx, j]) for j in range(Yn.shape[1])])
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xn = self._prep(X)
        P = np.stack([np.stack([m.predict(Xn) for m in bag], 1) for bag in self.bags])
        return P.mean(0) * self.ys + self.ym, P.std(0) * self.ys


MODELS = {"gp": GPSurrogate, "forest": ForestSurrogate, "boost": BoostSurrogate}
