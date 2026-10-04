import pytest

from sizeagent.stats import bootstrap_ci, mann_whitney_p, pairwise_pvalues, sims_to_spec, summarize, wilson_ci


def test_censoring():
    assert sims_to_spec(None, 300) == 301 and sims_to_spec(40, 300) == 40


def test_mann_whitney_separates_and_matches_identical():
    assert mann_whitney_p(range(1, 21), range(100, 121)) < 1e-4
    assert mann_whitney_p([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_cis_contain_estimate():
    lo, hi = bootstrap_ci([10, 12, 14, 16, 18])
    assert lo <= 14 <= hi
    lo, hi = wilson_ci(6, 6)
    assert 0.5 < lo < 1.0 and hi == pytest.approx(1.0)


def test_summarize():
    runs = [{"algo": a, "first_feasible": f} for a, f in
            [("x", 10), ("x", 20), ("x", 15), ("y", None), ("y", None), ("y", 250)]]
    s = summarize(runs, 300)
    assert s["x"]["success"] == 3 and s["y"]["median_sims"] == 301
    assert 0 <= pairwise_pvalues(s)["x vs y"] <= 1


def test_robust_stats_summarize():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rs", Path(__file__).parent.parent / "scripts" / "robust_stats.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = [{"algo": "a", "budget": 100, "first_robust_feasible": f, "full_pass": p, "full_total": 15,
             "best_metrics": {"power_uw": w}} for f, p, w in [(10, 15, 80.0), (20, 14, 90.0), (None, 5, 200.0)]]
    s = m.summarize(rows)["a"]
    assert s["robust_feasible"] == 2 and s["full_grid_all_pass"] == 1 and s["median_sims"] == 20
    assert s["median_power_uw"] == 85.0
