"""Corner-aware (robust) evaluation.

Hierarchical scheme: a design is first simulated at nominal (tt, 27 C). Only a
nominally feasible design is promoted to 6 extreme corners, and only if those
pass is it checked on the rest of the 15-point PVT grid, so the extra
simulations are spent where they can change the answer. Every corner run is
charged to the same budget as any other SPICE call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .circuit import Design
from .pvt import CORNERS, TEMPS
from .specs import BudgetExceeded, Evaluator, cost, violations
from .spice import SimResult, simulate

# Extremes that bound the PVT space: slow/fast at both temperature ends, plus skew corners.
ROBUST_CORNERS = [("ss", 125.0), ("ss", -40.0), ("ff", 125.0), ("ff", -40.0), ("fs", 27.0), ("sf", 27.0)]
# Tier 2: the remaining points of the full 5 corners x 3 temperatures grid (nominal tt/27 is tier 0).
GRID_REST = [(c, t) for c in CORNERS for t in TEMPS
             if (c, t) not in ROBUST_CORNERS and (c, t) != ("tt", 27.0)]


@dataclass
class RobustEvaluator(Evaluator):
    corners: list = field(default_factory=lambda: list(ROBUST_CORNERS))
    rest: list = field(default_factory=lambda: list(GRID_REST))   # only run if all of `corners` pass

    def _sweep(self, d: Design, points: list) -> float:
        total = 0.0
        for corner, temp in points:
            rc = simulate(d, corner, temp)
            self.sims += 1
            mc = rc.metrics() if rc.ok else None
            total += sum(violations(mc, self.spec).values()) if mc else 100.0
        return total

    def _promote(self, d: Design) -> float | None:
        """Total violation over the corner tiers, or None if the budget cannot cover them."""
        if self.sims + len(self.corners) > self.budget:
            return None
        total = self._sweep(d, self.corners)
        if total == 0 and self.rest:
            if self.sims + len(self.rest) > self.budget:
                return None
            total = self._sweep(d, self.rest)
        return total

    def __call__(self, d: Design) -> tuple[float, dict | None, SimResult | None]:
        k = d.key()
        if k in self.cache:
            return self.cache[k]
        if self.sims >= self.budget:
            raise BudgetExceeded
        r = simulate(d, self.corner, self.temp)
        self.sims += 1
        m = r.metrics() if r.ok else None
        c_nom = cost(m, self.spec)
        robust_ok, c, unverified = False, c_nom, False
        if m is not None and c_nom <= 0.1:
            total = self._promote(d)
            if total is None:
                # Budget ended mid-verification: keep the sims already charged in the trace as
                # an unverified design instead of silently dropping them.
                unverified, c = True, 0.55
            elif total == 0:
                robust_ok, c = True, c_nom        # in [0, 0.1]
            else:
                c = 0.2 + 0.7 * total / (1.0 + total)   # between nominal-feasible and infeasible
        best = min([row["cost"] for row in self.trace] + [c])
        self.trace.append({"sim": self.sims, "cost": c, "best_cost": best, "feasible": robust_ok,
                           "nominal_feasible": c_nom <= 0.1, "unverified": unverified, "metrics": m, "design": d.to_dict(),
                           "t": round(time.time() - self.t0, 2)})
        self.cache[k] = (c, m, r)
        return self.cache[k]
