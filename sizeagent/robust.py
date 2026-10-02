"""Corner-aware (robust) evaluation.

Hierarchical scheme: a design is first simulated at nominal (tt, 27 C). Only a
nominally feasible design is promoted to the corner set, so the extra
simulations are spent where they can change the answer. Every corner run is
charged to the same budget as any other SPICE call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .circuit import Design
from .specs import BudgetExceeded, Evaluator, cost, violations
from .spice import SimResult, simulate

# Extremes that bound the PVT space: slow/fast at both temperature ends, plus skew corners.
ROBUST_CORNERS = [("ss", 125.0), ("ss", -40.0), ("ff", 125.0), ("ff", -40.0), ("fs", 27.0), ("sf", 27.0)]


@dataclass
class RobustEvaluator(Evaluator):
    corners: list = field(default_factory=lambda: list(ROBUST_CORNERS))

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
        robust_ok, c = False, c_nom
        if m is not None and c_nom <= 0.1:
            if self.sims + len(self.corners) > self.budget:
                raise BudgetExceeded
            total = 0.0
            for corner, temp in self.corners:
                rc = simulate(d, corner, temp)
                self.sims += 1
                mc = rc.metrics() if rc.ok else None
                total += sum(violations(mc, self.spec).values()) if mc else 100.0
            if total == 0:
                robust_ok, c = True, c_nom        # in [0, 0.1]
            else:
                c = 0.2 + 0.7 * total / (1.0 + total)   # between nominal-feasible and infeasible
        best = min([row["cost"] for row in self.trace] + [c])
        self.trace.append({"sim": self.sims, "cost": c, "best_cost": best, "feasible": robust_ok,
                           "nominal_feasible": c_nom <= 0.1, "metrics": m, "design": d.to_dict(),
                           "t": round(time.time() - self.t0, 2)})
        self.cache[k] = (c, m, r)
        return self.cache[k]
