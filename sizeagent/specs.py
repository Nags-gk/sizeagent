"""Target specification, normalized violation, and a cached, budgeted evaluator."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .circuit import Design
from .spice import SimResult, simulate


@dataclass(frozen=True)
class Spec:
    gain_db: float = 60.0        # >=
    ugbw_mhz: float = 20.0       # >=
    pm_deg: float = 60.0         # >=
    power_uw: float = 250.0      # <=
    sat_margin_mv: float = 50.0  # every device: |Vds|-|Vdsat| >=
    vout_err_mv: float = 50.0    # |Vout_dc - Vcm| <= (systematic offset proxy)

    def describe(self) -> str:
        return (f"DC gain >= {self.gain_db} dB, UGBW >= {self.ugbw_mhz} MHz, phase margin >= "
                f"{self.pm_deg} deg, power <= {self.power_uw} uW, every transistor saturated with "
                f">= {self.sat_margin_mv} mV margin, |Vout_dc - 0.9 V| <= {self.vout_err_mv} mV")


def violations(m: dict, spec: Spec) -> dict:
    """Per-spec normalized shortfall (0 = met). Larger is worse."""
    def nz(x, default):
        return default if x is None or (isinstance(x, float) and math.isnan(x)) else x
    gain = nz(m.get("gain_db"), -100.0)
    ugbw = max(nz(m.get("ugbw_mhz"), 0.0), 1e-3)
    pm = nz(m.get("pm_deg"), -90.0)
    pwr = nz(m.get("power_uw"), 1e4)
    sat = nz(m.get("min_sat_margin_mv"), -1000.0)
    vout = abs(nz(m.get("vout_dc"), 0.0) - 0.9) * 1e3
    return {
        "gain_db": max(0.0, (spec.gain_db - gain) / 10.0),
        "ugbw_mhz": max(0.0, math.log10(spec.ugbw_mhz / ugbw)),
        "pm_deg": max(0.0, (spec.pm_deg - pm) / 10.0),
        "power_uw": max(0.0, math.log10(pwr / spec.power_uw)),
        "sat_margin_mv": max(0.0, (spec.sat_margin_mv - sat) / 50.0),
        "vout_err_mv": max(0.0, (vout - spec.vout_err_mv) / 50.0),
    }


def cost(m: dict | None, spec: Spec) -> float:
    """Total violation; feasible designs score in [0, 0.1] by power (lower is better)."""
    if m is None:
        return 100.0
    v = sum(violations(m, spec).values())
    return v + 0.1 * min(m["power_uw"], spec.power_uw) / spec.power_uw if v == 0 else 1.0 + v


def feasible(m: dict | None, spec: Spec) -> bool:
    return m is not None and sum(violations(m, spec).values()) == 0


class BudgetExceeded(Exception):
    pass


@dataclass
class Evaluator:
    """Counts unique SPICE runs, caches by design, and records the search trace."""
    spec: Spec = field(default_factory=Spec)
    budget: int = 200
    corner: str = "tt"
    temp: float = 27.0
    cache: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)   # one row per unique simulation
    sims: int = 0
    t0: float = field(default_factory=time.time)

    def __call__(self, d: Design) -> tuple[float, dict | None, SimResult | None]:
        k = d.key()
        if k in self.cache:
            return self.cache[k]
        if self.sims >= self.budget:
            raise BudgetExceeded
        r = simulate(d, self.corner, self.temp)
        self.sims += 1
        m = r.metrics() if r.ok else None
        c = cost(m, self.spec)
        best = min([row["cost"] for row in self.trace] + [c])
        self.trace.append({"sim": self.sims, "cost": c, "best_cost": best, "feasible": feasible(m, self.spec),
                           "metrics": m, "design": d.to_dict(), "t": round(time.time() - self.t0, 2)})
        self.cache[k] = (c, m, r)
        return self.cache[k]

    def first_feasible(self) -> int | None:
        return next((row["sim"] for row in self.trace if row["feasible"]), None)

    def best(self) -> dict | None:
        return min(self.trace, key=lambda r: r["cost"]) if self.trace else None
