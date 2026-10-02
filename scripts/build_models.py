"""Build trimmed ngspice corner libraries for sky130_fd_pr__{n,p}fet_01v8.

Each SkyWater corner file (e.g. *__tt.corner.spice) includes a corner-specific
model file (*__tt.pm3.spice). We copy both and patch the model subcircuit the
way open_pdks does for ngspice:
  * move the in-subckt `.param` defaults onto the `.subckt` line, and
  * pass the `mult` instance parameter through to the BSIM4 device as `m`.
"""
import csv, json, re, sys
from pathlib import Path

DEVICES = ["nfet_01v8", "pfet_01v8"]
CORNERS = ["tt", "ff", "ss", "fs", "sf"]


def patch_model(text: str, dev: str) -> str:
    hdr = re.compile(r"(\.subckt\s+sky130_fd_pr__" + dev + r"\s+d g s b)\s*\n\+\s*\n\.param\s+([^\n]+)\n")
    text, n = hdr.subn(lambda m: f"{m.group(1)} {m.group(2)}\n", text)
    if n != 1:
        sys.exit(f"unexpected subckt header for {dev}")
    inst = re.compile(r"^(msky130_fd_pr__" + dev + r" d g s b [^\n]*nf = \{nf\})\s*$", re.M)
    text, n = inst.subn(r"\1 m = {mult}", text)
    if n != 1:
        sys.exit(f"could not patch mult for {dev}")
    # Per-instance Monte Carlo mismatch (as open_pdks does): each *_slope_spectre
    # becomes a per-instance N(0,1) draw gated by mc_mm_switch.
    spec = re.compile(r"^\.param\s+(sky130_fd_pr__" + dev + r"__\w+_slope_spectre)\s*=\s*0\.0\s*$", re.M)
    names = spec.findall(text)
    if not names:
        sys.exit(f"no mismatch parameters found for {dev}")
    text = spec.sub("", text)
    local = "".join(f".param {n} = {{mc_mm_switch*agauss(0,1.0,1)}}\n" for n in names)
    text = re.sub(r"(\.subckt\s+sky130_fd_pr__" + dev + r"\s+d g s b[^\n]*\n)", lambda m: m.group(1) + local, text, count=1)
    return text


def main(src: str, out: str) -> None:
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bins = {}
    for dev in DEVICES:
        cdir = src / "cells" / dev
        with open(cdir / f"sky130_fd_pr__{dev}.bins.csv") as f:
            bins[dev] = sorted({(float(r["W"]), float(r["L"])) for r in csv.DictReader(f)})
        for c in CORNERS:
            pm3 = f"sky130_fd_pr__{dev}__{c}.pm3.spice"
            (out / pm3).write_text(patch_model((cdir / pm3).read_text(), dev))
            corner = (cdir / f"sky130_fd_pr__{dev}__{c}.corner.spice").read_text()
            corner, n = re.subn(r'\.include\s+"' + re.escape(pm3) + '"', f'.include "{(out / pm3).resolve()}"', corner)
            if n != 1:
                sys.exit(f"corner {c} of {dev} does not include {pm3}")
            (out / f"sky130_fd_pr__{dev}__{c}.corner.spice").write_text(corner)
    for c in CORNERS:
        # mc_mm_switch / mc_pr_switch are set by the testbench before including this file.
        lines = [f"* Trimmed SKY130 core-FET library, corner {c}",
                 f'.include "{(src / "models/parameters/lod.spice").resolve()}"',
                 ".param sky130_fd_pr__nfet_01v8__dlc_rotweak=0 sky130_fd_pr__pfet_01v8__dlc_rotweak=0"]
        for dev in DEVICES:
            lines.append(f'.include "{(src / f"cells/{dev}/sky130_fd_pr__{dev}__mismatch.corner.spice").resolve()}"')
            lines.append(f'.include "{(out / f"sky130_fd_pr__{dev}__{c}.corner.spice").resolve()}"')
        (out / f"sky130_core_{c}.spice").write_text("\n".join(lines) + "\n")
    (out / "bins.json").write_text(json.dumps(bins))
    print(f"built {len(CORNERS)} corner libs in {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
