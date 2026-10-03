"""Two-stage Miller-compensated op-amp on SKY130: design space and netlist.

Sizing is a discrete (combinatorial) problem: SKY130 transistor models are
binned, so each device group picks a characterized (W, L) pair per finger plus
an integer multiplier. Passives and the bias current come from fixed grids.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PDK_DIR = Path(os.environ.get("SIZEAGENT_PDK") or Path(__file__).resolve().parent.parent / "pdk_models")


class SetupError(RuntimeError):
    """The simulator or the SKY130 model libraries are not installed."""

# Device groups. Matched devices share one geometry.
#   inp  : M1/M2 NMOS differential pair
#   load : M3/M4 PMOS current-mirror load
#   bias : M8 (diode), M5 (tail), M7 (2nd-stage sink) NMOS mirror
#   cs   : M6 PMOS common-source 2nd stage
GROUPS = {"inp": "nfet_01v8", "load": "pfet_01v8", "bias": "nfet_01v8", "cs": "pfet_01v8"}
MULT_CHOICES = list(range(1, 17))
CC_PF = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
RZ_KOHM = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
IB_UA = [2, 3, 5, 7, 10, 15, 20, 30]
CL_PF = 2.0
VDD = 1.8
VCM = 0.9


@lru_cache(maxsize=1)
def valid_geometries() -> dict[str, list[tuple[float, float]]]:
    """Characterized (W_um, L_um) per-finger pairs for each device type."""
    bins = PDK_DIR / "bins.json"
    if not bins.exists():
        raise SetupError(f"SKY130 models not found at {PDK_DIR}. Run ./scripts/setup_pdk.sh "
                         "(or point SIZEAGENT_PDK at a built pdk_models directory).")
    data = json.loads(bins.read_text())
    # Restrict to analog-friendly sizes: skip minimum-width fingers below 0.5 um.
    return {dev: [tuple(p) for p in pairs if p[0] >= 0.5] for dev, pairs in data.items()}


@dataclass(frozen=True)
class Design:
    """One point in the sizing space. Geometries are indices into valid_geometries()."""
    geo: dict = field(default_factory=dict)   # group -> index into valid geometries
    mult: dict = field(default_factory=dict)  # "m1","m3","m5","m6","m7" -> int
    cc: float = 1.0                           # pF
    rz: float = 0.0                           # kOhm
    ib: float = 10.0                          # uA

    def wl(self, group: str) -> tuple[float, float]:
        return valid_geometries()[GROUPS[group]][self.geo[group]]

    def to_dict(self) -> dict:
        d = {}
        for g in GROUPS:
            w, l = self.wl(g)
            d[f"{g}_w"], d[f"{g}_l"] = w, l
        d.update(self.mult)
        d.update(cc_pf=self.cc, rz_kohm=self.rz, ib_ua=self.ib)
        return d

    def key(self) -> tuple:
        return (tuple(sorted(self.geo.items())), tuple(sorted(self.mult.items())), self.cc, self.rz, self.ib)


MULT_KEYS = ["m1", "m3", "m5", "m6", "m7"]


def space_size() -> int:
    geos = valid_geometries()
    n = 1
    for g in GROUPS:
        n *= len(geos[GROUPS[g]])
    return n * len(MULT_CHOICES) ** len(MULT_KEYS) * len(CC_PF) * len(RZ_KOHM) * len(IB_UA)


def random_design(rng: random.Random) -> Design:
    geos = valid_geometries()
    return Design(
        geo={g: rng.randrange(len(geos[GROUPS[g]])) for g in GROUPS},
        mult={k: rng.choice(MULT_CHOICES) for k in MULT_KEYS},
        cc=rng.choice(CC_PF), rz=rng.choice(RZ_KOHM), ib=rng.choice(IB_UA),
    )


def snap(group_or_key: str, value: float) -> float:
    """Snap a value to the nearest allowed grid point (used by the LLM agent)."""
    grid = {"cc": CC_PF, "rz": RZ_KOHM, "ib": IB_UA}.get(group_or_key, MULT_CHOICES)
    return min(grid, key=lambda g: abs(g - value))


def nearest_geometry(group: str, w: float, l: float) -> int:
    """Index of the characterized geometry closest to (w, l) in log space."""
    import math
    pairs = valid_geometries()[GROUPS[group]]
    return min(range(len(pairs)),
               key=lambda i: (math.log(pairs[i][0] / w)) ** 2 + 4 * (math.log(pairs[i][1] / l)) ** 2)


def netlist(d: Design, corner: str = "tt", temp: float = 27.0, mc: bool = False) -> str:
    """Open-loop AC testbench. A 1 GH inductor closes a unity-gain loop at DC to
    set the operating point, and opens it for the AC sweep."""
    def fet(name, dn, gn, sn, bn, group, m):
        w, l = d.wl(group)
        return f"X{name} {dn} {gn} {sn} {bn} sky130_fd_pr__{GROUPS[group]} W={w} L={l} mult={m}"

    lib = (PDK_DIR / f"sky130_core_{corner}.spice").resolve()
    lines = [
        "* two-stage miller op-amp, sky130",
        # SKY130 models expect W/L in microns with a 1e-6 scale (needed for the
        # mismatch terms, which divide by sqrt(W*L*mult) in um^2).
        ".option scale=1e-6",
        f".param mc_mm_switch={1 if mc else 0} mc_pr_switch=0",
        f'.include "{lib}"',
        f".temp {temp}",
        f"VDD vdd 0 {VDD}",
        f"IREF vdd nbias {d.ib}u",
        fet("M8", "nbias", "nbias", "0", "0", "bias", 1),
        fet("M5", "tail", "nbias", "0", "0", "bias", d.mult["m5"]),
        fet("M1", "x1", "inp", "tail", "0", "inp", d.mult["m1"]),
        fet("M2", "out1", "inn", "tail", "0", "inp", d.mult["m1"]),
        fet("M3", "x1", "x1", "vdd", "vdd", "load", d.mult["m3"]),
        fet("M4", "out1", "x1", "vdd", "vdd", "load", d.mult["m3"]),
        fet("M6", "out", "out1", "vdd", "vdd", "cs", d.mult["m6"]),
        fet("M7", "out", "nbias", "0", "0", "bias", d.mult["m7"]),
        f"RZ out1 cz {max(d.rz, 1e-3)}k",
        f"CC cz out {d.cc}p",
        f"CL out 0 {CL_PF}p",
        # M2 gate (inn) is the non-inverting input; feedback goes to M1 gate (inp).
        f"VIN inn 0 dc {VCM} ac 1",
        "LFB out inp 1G",
        "CFB inp 0 1G",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def multiplicity(d: Design) -> dict:
    return {"M1": d.mult["m1"], "M2": d.mult["m1"], "M3": d.mult["m3"], "M4": d.mult["m3"],
            "M5": d.mult["m5"], "M6": d.mult["m6"], "M7": d.mult["m7"], "M8": 1}


DEVICES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
DEVICE_TYPE = {"M1": "nfet_01v8", "M2": "nfet_01v8", "M3": "pfet_01v8", "M4": "pfet_01v8",
               "M5": "nfet_01v8", "M6": "pfet_01v8", "M7": "nfet_01v8", "M8": "nfet_01v8"}


def reference_design() -> Design:
    """A hand-sized starting point (meets gain/UGBW/PM but not the saturation margin)."""
    return Design(geo={"inp": nearest_geometry("inp", 3, 0.5), "load": nearest_geometry("load", 3, 1),
                       "bias": nearest_geometry("bias", 3, 1), "cs": nearest_geometry("cs", 5, 0.5)},
                  mult={"m1": 4, "m3": 2, "m5": 4, "m6": 8, "m7": 8}, cc=1.0, rz=2.0, ib=10)
