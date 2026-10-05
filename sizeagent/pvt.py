"""Process/voltage/temperature corner sweep and Monte Carlo mismatch analysis."""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .circuit import Design
from .specs import Spec, violations
from .spice import simulate

CORNERS = ["tt", "ff", "ss", "fs", "sf"]
TEMPS = [-40.0, 27.0, 125.0]


def corner_sweep(d: Design, spec: Spec, workers: int = 2) -> dict:
    jobs = [(c, t) for c in CORNERS for t in TEMPS]
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda ct: simulate(d, corner=ct[0], temp=ct[1]), jobs))
    rows: list[dict[str, Any]] = []
    for (c, t), r in zip(jobs, results):
        m = r.metrics() if r.ok else None
        v: dict[str, float] = violations(m, spec) if m else {}
        rows.append({"corner": c, "temp_c": t, "metrics": m,
                     "pass": bool(m) and sum(v.values()) == 0,
                     "failing": [k for k, x in v.items() if x > 0] if m else ["sim_failed"]})
    ok: list[dict[str, float]] = [r["metrics"] for r in rows if r["metrics"]]
    worst = {
        "gain_db": min(m["gain_db"] for m in ok), "ugbw_mhz": min(m["ugbw_mhz"] for m in ok),
        "pm_deg": min(m["pm_deg"] for m in ok), "power_uw": max(m["power_uw"] for m in ok),
        "min_sat_margin_mv": min(m["min_sat_margin_mv"] for m in ok),
    } if ok else {}
    return {"rows": rows, "worst": worst, "pass_count": sum(r["pass"] for r in rows), "total": len(rows)}


def monte_carlo(d: Design, n: int = 50, workers: int = 2) -> dict:
    """Mismatch-only Monte Carlo at tt/27C. Input-referred offset ~ Vout shift in
    the unity-gain DC loop."""
    with ThreadPoolExecutor(workers) as ex:
        rs = list(ex.map(lambda s: simulate(d, mc=True, seed=s), range(1, n + 1)))
    vos = [(r.vout_dc - 0.9) * 1e3 for r in rs if r.ok]
    gains = [r.gain_db for r in rs if r.ok]
    pms = [r.pm_deg for r in rs if r.ok]
    return {"n": len(vos), "offset_mv": vos, "offset_sigma_mv": statistics.pstdev(vos) if vos else None,
            "gain_db_min": min(gains) if gains else None, "pm_deg_min": min(pms) if pms else None}
