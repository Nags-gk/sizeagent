"""Optimizer, surrogate and PVT logic against the analytic fake simulator."""
import numpy as np
import pytest
from fake_sim import failing_simulate, fake_simulate

import sizeagent.pvt as pvt
import sizeagent.robust as robust
import sizeagent.specs as specs
from sizeagent.circuit import reference_design
from sizeagent.optimizers.search import ALGORITHMS, local_search
from sizeagent.specs import Evaluator, Spec


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    monkeypatch.setattr(specs, "simulate", fake_simulate)
    monkeypatch.setattr(pvt, "simulate", fake_simulate)
    monkeypatch.setattr(robust, "simulate", fake_simulate)


@pytest.mark.parametrize("algo", ["random", "ga", "sa", "tpe"])
def test_algorithms_use_exactly_the_budget_and_are_reproducible(algo):
    a, b = Evaluator(budget=40), Evaluator(budget=40)
    ALGORITHMS[algo](a, seed=3)
    ALGORITHMS[algo](b, seed=3)
    assert a.sims == 40 and len(a.trace) == 40
    assert [r["cost"] for r in a.trace] == [r["cost"] for r in b.trace]
    assert all(x["best_cost"] <= y["best_cost"] for y, x in zip(a.trace, a.trace[1:]))


def test_surrogate_ga_runs_and_improves_on_warmup():
    ev = Evaluator(budget=60)
    ALGORITHMS["surrogate_ga"](ev, seed=0, warmup=20, per_gen=5, pool=60, epochs=15)
    assert ev.sims == 60
    assert ev.trace[-1]["best_cost"] <= ev.trace[19]["best_cost"]


def test_surrogate_ga_survives_failing_simulator(monkeypatch):
    monkeypatch.setattr(specs, "simulate", failing_simulate)
    ev = Evaluator(budget=30)
    ALGORITHMS["surrogate_ga"](ev, seed=0, warmup=5, per_gen=3, pool=30, epochs=5)
    assert ev.sims == 30 and not any(r["feasible"] for r in ev.trace)


def test_surrogate_fit_predict_shapes_and_uncertainty():
    from sizeagent.surrogate import Surrogate
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    Y = np.stack([X[:, 0] * 2 + X[:, 1], np.sin(X[:, 2])], 1)
    s = Surrogate(n_models=3, hidden=32, epochs=60).fit(X, Y)
    mu, sd = s.predict(X[:10])
    assert mu.shape == (10, 2) and sd.shape == (10, 2) and (sd >= 0).all()
    assert np.abs(mu[:, 0] - Y[:10, 0]).mean() < 0.5


def test_local_search_never_exceeds_requested_sims():
    from sizeagent.optimizers import encode
    ev = Evaluator(budget=100)
    local_search(ev, encode(reference_design()), 8)
    assert ev.sims <= 8


def test_corner_sweep_and_worst_case():
    r = pvt.corner_sweep(reference_design(), Spec(), workers=1)
    assert r["total"] == 15 and len(r["rows"]) == 15
    assert r["worst"]["pm_deg"] <= min(x["metrics"]["pm_deg"] for x in r["rows"])


def test_corner_sweep_all_failing(monkeypatch):
    monkeypatch.setattr(pvt, "simulate", failing_simulate)
    r = pvt.corner_sweep(reference_design(), Spec(), workers=1)
    assert r["pass_count"] == 0 and r["worst"] == {}


def test_monte_carlo_reports_spread():
    r = pvt.monte_carlo(reference_design(), n=10, workers=1)
    assert r["n"] == 10 and r["offset_sigma_mv"] > 0


def test_robust_evaluator_never_drops_charged_sims():
    nominal = Evaluator(budget=300)
    ALGORITHMS["random"](nominal, seed=1)
    feas = next(r for r in nominal.trace if r["feasible"])
    from sizeagent.agent.tools import design_from_args
    d = design_from_args(feas["design"])
    ev = robust.RobustEvaluator(budget=3)               # nominal feasible, but 6 corners do not fit
    c, _, _ = ev(d)
    assert ev.sims == 1 and len(ev.trace) == 1 and ev.trace[0]["unverified"] and not ev.trace[0]["feasible"]
    assert 0.1 < c < 1.0
    ev2 = robust.RobustEvaluator(budget=200)
    ev2(d)
    assert ev2.sims >= 1 + len(ev2.corners) and not ev2.trace[0]["unverified"]
