import random

import pytest

from sizeagent.circuit import Design, random_design, reference_design, space_size
from sizeagent.optimizers import decode, encode
from sizeagent.specs import Evaluator, Spec, cost, feasible, violations
from sizeagent.spice import simulate


def test_reference_design_simulates_sanely():
    r = simulate(reference_design())
    assert r.ok, r.error
    m = r.metrics()
    assert 65 < m["gain_db"] < 85
    assert 20 < m["ugbw_mhz"] < 80
    assert 40 < m["pm_deg"] < 80
    assert abs(m["vout_dc"] - 0.9) < 0.02          # unity-gain DC loop settles at Vcm
    # Power = VDD * Ib * (1 + m5 + m7) = 1.8 V * 10 uA * 13 = 234 uW (minus mirror error)
    assert 200 < m["power_uw"] < 240


def test_multiplier_scales_current():
    d = reference_design()
    d2 = Design(geo=d.geo, mult={**d.mult, "m7": 16}, cc=d.cc, rz=d.rz, ib=d.ib)
    i1 = abs(simulate(d).op["M7"]["id"])
    i2 = abs(simulate(d2).op["M7"]["id"])
    assert i2 / i1 == pytest.approx(2.0, rel=0.05)


def test_corners_differ():
    d = reference_design()
    ss, ff = simulate(d, corner="ss"), simulate(d, corner="ff")
    assert ss.ok and ff.ok
    assert ff.ugbw_hz > ss.ugbw_hz


def test_monte_carlo_mismatch_varies_offset():
    d = reference_design()
    v = {round(simulate(d, mc=True, seed=s).vout_dc, 6) for s in (1, 2, 3)}
    assert len(v) == 3
    assert simulate(d, mc=False).vout_dc == simulate(d, mc=False, seed=7).vout_dc


def test_encode_decode_roundtrip():
    rng = random.Random(0)
    for _ in range(50):
        d = random_design(rng)
        assert decode(encode(d)).key() == d.key()
    assert space_size() > 1e15


def test_violation_and_cost():
    spec = Spec()
    good = {"gain_db": 70, "ugbw_mhz": 30, "pm_deg": 65, "power_uw": 100, "vout_dc": 0.9, "min_sat_margin_mv": 100}
    assert feasible(good, spec) and 0 <= cost(good, spec) <= 0.1
    bad = {**good, "pm_deg": 40}
    assert violations(bad, spec)["pm_deg"] == pytest.approx(2.0)
    assert cost(bad, spec) > 1 and cost(None, spec) == 100


def test_evaluator_caches_and_budgets():
    ev = Evaluator(budget=2)
    d = reference_design()
    ev(d)
    ev(d)
    assert ev.sims == 1
    ev(random_design(random.Random(1)))
    from sizeagent.specs import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        ev(random_design(random.Random(2)))


def test_robust_evaluator_promotes_only_nominal_feasible(monkeypatch):
    import sizeagent.robust as rb
    from sizeagent.robust import RobustEvaluator
    from sizeagent.spice import SimResult

    good = SimResult(ok=True, gain_db=70, ugbw_hz=30e6, pm_deg=65, power_uw=100, vout_dc=0.9,
                     op={"M1": {"vds": 0.5, "vdsat": 0.1}})
    bad_corner = SimResult(ok=True, gain_db=70, ugbw_hz=30e6, pm_deg=40, power_uw=100, vout_dc=0.9,
                           op={"M1": {"vds": 0.5, "vdsat": 0.1}})
    calls = []

    def fake(d, corner="tt", temp=27.0, **kw):
        calls.append((corner, temp))
        return good if (corner, temp) == ("tt", 27.0) or fail_at is None else bad_corner

    fail_at = None
    monkeypatch.setattr(rb, "simulate", fake)
    ev = RobustEvaluator(budget=50)
    c, _, _ = ev(random_design(random.Random(1)))
    assert len(calls) == 1 + len(ev.corners) + len(ev.rest) == 15 and ev.trace[-1]["feasible"] and c <= 0.1
    fail_at = ("ss", 125.0)
    c2, _, _ = ev(random_design(random.Random(2)))
    assert not ev.trace[-1]["feasible"] and ev.trace[-1]["nominal_feasible"] and 0.2 < c2 < 1.0
    assert ev.sims == 15 + 1 + len(ev.corners)
