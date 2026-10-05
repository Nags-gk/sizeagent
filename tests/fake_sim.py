"""Deterministic analytic stand-in for ngspice, so optimizer/surrogate/PVT logic
can be tested in milliseconds without the simulator or the PDK."""
import math

from sizeagent.spice import SimResult

CORNER_PM = {"tt": 0.0, "ff": 4.0, "ss": -6.0, "fs": -2.0, "sf": -2.0}


def fake_simulate(d, corner="tt", temp=27.0, mc=False, timeout=60.0, seed=1):
    lin, lld, lcs = d.wl("inp")[1], d.wl("load")[1], d.wl("cs")[1]
    gain = 38 + 9 * math.log2(lin / 0.15 + 1) + 4 * math.log2(lld + 1) + 3 * math.log2(lcs / 0.15 + 1)
    ugbw = 8 * d.mult["m1"] ** 0.5 * (d.ib / 5) ** 0.5 / (d.cc ** 0.7)
    pm = 70 - 4 * abs(math.log(d.mult["m6"] / (2 * d.mult["m1"]))) * 5 + CORNER_PM[corner] - 0.02 * (temp - 27) ** 1
    power = 1.8 * d.ib * (1 + d.mult["m5"] + d.mult["m7"])
    margin = 0.15 - 0.004 * abs(d.mult["m7"] - 2 * d.mult["m5"]) - 0.0005 * d.mult["m6"]
    vout = 0.9 + (0.002 * (d.mult["m6"] - 2 * d.mult["m7"] / max(d.mult["m5"], 1)) if not mc else 0.01 * (seed % 5 - 2))
    return SimResult(ok=True, gain_db=gain, ugbw_hz=ugbw * 1e6, pm_deg=pm, power_uw=power, vout_dc=vout,
                     op={"M1": {"vds": 0.5, "vdsat": 0.5 - margin, "id": 1e-5, "m": 1}})


def failing_simulate(d, corner="tt", temp=27.0, **kw):
    return SimResult(ok=False, error="simulation failed")
