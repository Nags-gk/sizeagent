"""Tools the sizing agent can call. Each returns a JSON-serializable dict."""
from __future__ import annotations

from ..circuit import (CC_PF, GROUPS, IB_UA, MULT_CHOICES, MULT_KEYS, RZ_KOHM, Design,
                       nearest_geometry, snap, valid_geometries)
from ..optimizers import decode, encode
from ..optimizers.search import local_search
from ..pvt import corner_sweep
from ..specs import BudgetExceeded, Evaluator, violations

DESIGN_SCHEMA = {
    "type": "object",
    "description": "Full op-amp sizing. W/L in um per finger (snapped to the nearest characterized "
                   "SKY130 geometry); m* are integer multipliers 1-16.",
    "properties": {
        **{f"{g}_{p}": {"type": "number"} for g in GROUPS for p in ("w", "l")},
        **{k: {"type": "integer"} for k in MULT_KEYS},
        "cc_pf": {"type": "number"}, "rz_kohm": {"type": "number"}, "ib_ua": {"type": "number"},
    },
    "required": [f"{g}_{p}" for g in GROUPS for p in ("w", "l")] + MULT_KEYS + ["cc_pf", "rz_kohm", "ib_ua"],
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "simulate",
        "description": "Run one ngspice simulation (costs 1 from the budget). Returns snapped design, "
                       "metrics, per-spec violations and each transistor's operating point.",
        "parameters": {"type": "object", "properties": {"design": DESIGN_SCHEMA}, "required": ["design"]}}},
    {"type": "function", "function": {
        "name": "list_geometries",
        "description": "List characterized (W, L) per-finger sizes in um for a device type. Free.",
        "parameters": {"type": "object", "properties": {"device": {"type": "string", "enum": ["nfet_01v8", "pfet_01v8"]}},
                       "required": ["device"]}}},
    {"type": "function", "function": {
        "name": "refine",
        "description": "Run a short combinatorial local search (simulated annealing) starting from a design. "
                       "Uses up to `sims` simulations (max 40). Returns the best design found.",
        "parameters": {"type": "object", "properties": {"design": DESIGN_SCHEMA, "sims": {"type": "integer"}},
                       "required": ["design", "sims"]}}},
    {"type": "function", "function": {
        "name": "verify_pvt",
        "description": "Check a design across 5 process corners x 3 temperatures (-40, 27, 125 C). "
                       "Does not use the search budget; limited to 3 calls.",
        "parameters": {"type": "object", "properties": {"design": DESIGN_SCHEMA}, "required": ["design"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Finish and submit the final design with a short rationale.",
        "parameters": {"type": "object", "properties": {"design": DESIGN_SCHEMA, "rationale": {"type": "string"}},
                       "required": ["design", "rationale"]}}},
]


def design_from_args(a: dict) -> Design:
    geo = {g: nearest_geometry(g, float(a[f"{g}_w"]), float(a[f"{g}_l"])) for g in GROUPS}
    mult = {k: int(snap(k, float(a[k]))) for k in MULT_KEYS}
    return Design(geo=geo, mult=mult, cc=snap("cc", float(a["cc_pf"])),
                  rz=snap("rz", float(a["rz_kohm"])), ib=snap("ib", float(a["ib_ua"])))


def _r(x, n=3):
    return round(x, n) if isinstance(x, float) else x


def op_summary(res) -> dict:
    out = {}
    for dev, p in res.op.items():
        if not {"id", "gm", "vds", "vdsat", "vgs", "vth"} <= p.keys():
            continue
        out[dev] = {"id_ua": _r(abs(p["id"]) * 1e6, 2),
                    "gm_over_id": _r(p["gm"] / abs(p["id"]), 1) if p["id"] else None,
                    "vov_mv": _r((abs(p["vgs"]) - abs(p["vth"])) * 1e3, 0),
                    "sat_margin_mv": _r((abs(p["vds"]) - abs(p["vdsat"])) * 1e3, 0),
                    "m": p.get("m")}
    return out


class AgentTools:
    def __init__(self, ev: Evaluator, max_pvt_calls: int = 3):
        self.ev, self.pvt_calls, self.max_pvt_calls = ev, 0, max_pvt_calls
        self.submitted: Design | None = None
        self.rationale = ""

    def _budget(self) -> dict:
        return {"sims_used": self.ev.sims, "sims_left": self.ev.budget - self.ev.sims}

    def simulate(self, design: dict) -> dict:
        d = design_from_args(design)
        try:
            c, m, res = self.ev(d)
        except BudgetExceeded:
            return {"error": "simulation budget exhausted; submit your best design", **self._budget()}
        if m is None:
            return {"design": d.to_dict(), "error": res.error or "simulation failed", **self._budget()}
        v = violations(m, self.ev.spec)
        return {"design": d.to_dict(), "metrics": {k: _r(x, 2) for k, x in m.items()},
                "violations": {k: _r(x) for k, x in v.items() if x > 0},
                "meets_spec": not any(x > 0 for x in v.values()),
                "operating_point": op_summary(res), **self._budget()}

    def list_geometries(self, device: str) -> dict:
        return {"device": device, "w_l_um": valid_geometries()[device],
                "multipliers": MULT_CHOICES, "cc_pf": CC_PF, "rz_kohm": RZ_KOHM, "ib_ua": IB_UA}

    def refine(self, design: dict, sims: int) -> dict:
        sims = max(1, min(int(sims), 40, self.ev.budget - self.ev.sims))
        c, best = local_search(self.ev, encode(design_from_args(design)), sims, seed=self.ev.sims)
        d = decode(best)
        _, m, _ = self.ev.cache.get(d.key(), (None, None, None))
        v = violations(m, self.ev.spec) if m else {}
        return {"best_design": d.to_dict(), "metrics": {k: _r(x, 2) for k, x in (m or {}).items()},
                "violations": {k: _r(x) for k, x in v.items() if x > 0},
                "meets_spec": bool(m) and not any(x > 0 for x in v.values()), **self._budget()}

    def verify_pvt(self, design: dict) -> dict:
        if self.pvt_calls >= self.max_pvt_calls:
            return {"error": "verify_pvt call limit reached"}
        self.pvt_calls += 1
        r = corner_sweep(design_from_args(design), self.ev.spec)
        return {"pass_count": r["pass_count"], "total": r["total"],
                "worst_case": {k: _r(x, 2) for k, x in r["worst"].items()},
                "failing_corners": [f'{x["corner"]}@{x["temp_c"]:.0f}C: {",".join(x["failing"])}'
                                    for x in r["rows"] if not x["pass"]]}

    def submit(self, design: dict, rationale: str) -> dict:
        self.submitted = design_from_args(design)
        self.rationale = rationale
        return {"ok": True}

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if name not in {t["function"]["name"] for t in TOOL_SPECS} or fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except (KeyError, TypeError, ValueError) as e:
            return {"error": f"bad arguments for {name}: {e}"}
