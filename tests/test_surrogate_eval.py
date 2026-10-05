import numpy as np
import pytest

from sizeagent.specs import Spec
from sizeagent.surrogate import targets
from sizeagent.surrogate_eval import auroc, calibration, feasibility_ranking, r2_mae, spearman


def test_r2_mae_perfect_and_mean_predictor():
    y = np.arange(10.0)
    assert r2_mae(y, y) == (1.0, 0.0)
    assert r2_mae(y, np.full(10, y.mean()))[0] == pytest.approx(0.0)


def test_spearman_monotone_and_ties():
    assert spearman(np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])) == pytest.approx(1.0)
    assert spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])) == pytest.approx(-1.0)
    assert spearman(np.ones(5), np.arange(5)) == 0.0


def test_calibration_of_a_calibrated_and_overconfident_model():
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    mu = np.zeros(5000)
    good = calibration(y, mu, np.ones(5000))
    assert 0.88 < good["coverage90"] < 0.92
    assert calibration(y, mu, np.full(5000, 0.2))["coverage90"] < 0.4


def test_auroc_direction():
    score = np.array([0.0, 0.1, 0.9, 1.0])
    assert auroc(score, np.array([True, True, False, False])) == 1.0
    assert auroc(score, np.array([False, False, True, True])) == 0.0


def test_feasibility_ranking_perfect_predictor():
    good = {"gain_db": 70, "ugbw_mhz": 30, "pm_deg": 65, "power_uw": 100, "vout_dc": 0.9, "min_sat_margin_mv": 100}
    bad = {**good, "pm_deg": 30}
    Y = np.array([targets(m) for m in [good, bad, good, bad, bad]])
    r = feasibility_ranking(Y, Y, Spec(), ks=(2,))
    assert r["auroc"] == 1.0 and r["precision_at_2"] == 1.0 and r["n_feasible"] == 2


def test_conformal_calibration_fixes_overconfident_sd():
    from sizeagent.surrogate import CalibratedSurrogate

    class Overconfident:
        ys = np.ones(1)

        def fit(self, X, Y):
            self.m = Y.mean(0)
            return self

        def predict(self, X):
            return np.tile(self.m, (len(X), 1)), np.full((len(X), 1), 0.1)   # true noise sd is 1

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    Y = rng.normal(size=(2000, 1))
    s = CalibratedSurrogate(Overconfident, seed=0).fit(X[:1500], Y[:1500])
    mu, sd = s.predict(X[1500:])
    cov = calibration(Y[1500:, 0], mu[:, 0], sd[:, 0])["coverage90"]
    assert 0.85 < cov < 0.95
    raw = Overconfident().fit(X, Y).predict(X[1500:])
    assert calibration(Y[1500:, 0], raw[0][:, 0], raw[1][:, 0])["coverage90"] < 0.2
