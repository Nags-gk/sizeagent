"""ngspice runner: simulate a Design and return measured performance."""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .circuit import DEVICE_TYPE, DEVICES, VDD, Design, SetupError, multiplicity, netlist

OP_PARAMS = ["id", "gm", "gds", "vth", "vdsat", "vds", "vgs"]


@dataclass
class SimResult:
    ok: bool
    gain_db: float = float("nan")
    ugbw_hz: float = float("nan")
    pm_deg: float = float("nan")
    power_uw: float = float("nan")
    vout_dc: float = float("nan")
    op: dict = field(default_factory=dict)   # device -> {param: value}
    error: str = ""

    def metrics(self) -> dict:
        return {"gain_db": self.gain_db, "ugbw_mhz": self.ugbw_hz / 1e6, "pm_deg": self.pm_deg,
                "power_uw": self.power_uw, "vout_dc": self.vout_dc,
                "min_sat_margin_mv": self.min_sat_margin_mv()}

    def sat_margins(self) -> dict:
        """|Vds| - |Vdsat| per device in mV; negative means the device is in triode."""
        return {d: 1e3 * (abs(p["vds"]) - abs(p["vdsat"])) for d, p in self.op.items()
                if "vds" in p and "vdsat" in p}

    def min_sat_margin_mv(self) -> float:
        m = self.sat_margins()
        return min(m.values()) if m else float("nan")


def _control_block() -> str:
    prints = []
    for d in DEVICES:
        dev = f"@m.x{d.lower()}.msky130_fd_pr__{DEVICE_TYPE[d]}"
        for p in OP_PARAMS:
            prints.append(f"print {dev}[{p}]")
    return "\n".join([
        ".option itl1=500",
        ".control",
        # The AC analysis computes the DC operating point once; device OP values
        # remain readable afterwards, so no separate `op` run is needed.
        "ac dec 40 1 10G",
        *prints,
        "let gdb = db(v(out))",
        "let ph = 180/pi*cph(v(out))",
        "meas ac gain_db find gdb at=1",
        "meas ac ugbw when gdb=0 fall=1",
        "meas ac ph_ugbw find ph when gdb=0 fall=1",
        ".endc",
    ])


_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _parse(out: str, mult: dict, ib_ua: float) -> SimResult:
    r = SimResult(ok=False)
    vals = {}
    for m in re.finditer(r"^\s*([@\w.\[\]()]+)\s*=\s*" + _NUM, out, re.M):
        vals[m.group(1).lower()] = float(m.group(2))
    try:
        for d in DEVICES:
            dev = f"@m.x{d.lower()}.msky130_fd_pr__{DEVICE_TYPE[d]}"
            r.op[d] = {p: vals[f"{dev}[{p}]"] for p in OP_PARAMS if f"{dev}[{p}]" in vals}
            # id, gm, gds are totals for the device including its multiplier.
            r.op[d]["m"] = mult[d]
        # All supply current flows through Iref, M5 (tail) and M7 (2nd-stage sink).
        r.power_uw = VDD * (ib_ua * 1e-6 + abs(r.op["M5"]["id"]) + abs(r.op["M7"]["id"])) * 1e6
        r.vout_dc = r.op["M7"]["vds"]                     # M7 source is grounded
        r.gain_db = vals["gain_db"]
    except KeyError as e:
        r.error = f"missing measurement {e}"
        return r
    r.ugbw_hz = vals.get("ugbw", float("nan"))
    ph = vals.get("ph_ugbw", float("nan"))
    r.pm_deg = 180.0 + ph if not math.isnan(ph) else float("nan")
    if r.gain_db <= 0:
        r.ugbw_hz, r.pm_deg = 0.0, 0.0
    r.ok = not math.isnan(r.gain_db)
    return r


def simulate(d: Design, corner: str = "tt", temp: float = 27.0, mc: bool = False,
             timeout: float = 60.0, seed: int = 1) -> SimResult:
    if shutil.which("ngspice") is None:
        raise SetupError("ngspice not found on PATH (macOS: brew install ngspice; Debian/Ubuntu: apt install ngspice)")
    net = netlist(d, corner, temp, mc).replace(".end\n", _control_block() + "\n.end\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tb.sp"
        path.write_text(net)
        (Path(tmp) / ".spiceinit").write_text(f"set rndseed={seed}\n")
        try:
            proc = subprocess.run(["ngspice", "-b", str(path)], capture_output=True, text=True,
                                  timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return SimResult(ok=False, error="timeout")
    res = _parse(proc.stdout + proc.stderr, multiplicity(d), d.ib)
    if not res.ok and not res.error:
        res.error = "simulation failed"
    return res
