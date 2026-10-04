"""Statistics for the corner-aware study (results/robust_full_20.jsonl by default).

Reports, per algorithm: robust-feasible rate (Wilson CI), median simulations to a
robust-feasible design (bootstrap CI, failures censored at budget + 1), how often the final
design passes the whole 15-point PVT grid, and pairwise Mann-Whitney tests.
Writes results/robust_stats.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sizeagent.stats import bootstrap_ci, mann_whitney_p, sims_to_spec, wilson_ci  # noqa: E402


def summarize(rows: list[dict]) -> dict:
    out = {}
    for algo in sorted({r["algo"] for r in rows}):
        rs = [r for r in rows if r["algo"] == algo]
        budget = rs[0]["budget"]
        sims = [sims_to_spec(r["first_robust_feasible"], budget) for r in rs]
        ok = sum(r["first_robust_feasible"] is not None for r in rs)
        full = sum(r["full_pass"] == r["full_total"] for r in rs)
        out[algo] = {
            "n": len(rs), "budget": budget,
            "robust_feasible": ok, "robust_feasible_ci": wilson_ci(ok, len(rs)),
            "median_sims": float(np.median(sims)), "median_sims_ci": bootstrap_ci(sims), "sims": sims,
            "full_grid_all_pass": full, "full_grid_all_pass_ci": wilson_ci(full, len(rs)),
            "mean_grid_pass": float(np.mean([r["full_pass"] for r in rs])), "full_total": rs[0]["full_total"],
            "median_power_uw": float(np.median([r["best_metrics"]["power_uw"] for r in rs
                                                if r["first_robust_feasible"]])) if ok else None,
        }
    return out


def main(path: str = "results/robust_full_20.jsonl") -> None:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    s = summarize(rows)
    print("| Strategy | Robust-feasible (95% CI) | Median sims (95% CI) | Final design passes all grid points | Mean grid points passed |")
    print("|---|---|---|---|---|")
    for a, v in s.items():
        r0, r1 = v["robust_feasible_ci"]
        m0, m1 = v["median_sims_ci"]
        f0, f1 = v["full_grid_all_pass_ci"]
        print(f"| {a} | {v['robust_feasible']}/{v['n']} ({r0:.0%}-{r1:.0%}) | {v['median_sims']:.0f} ({m0:.0f}-{m1:.0f}) | "
              f"{v['full_grid_all_pass']}/{v['n']} ({f0:.0%}-{f1:.0%}) | {v['mean_grid_pass']:.1f}/{v['full_total']} |")
    algos = list(s)
    p = {f"{a} vs {b}": mann_whitney_p(s[a]["sims"], s[b]["sims"]) for i, a in enumerate(algos) for b in algos[i + 1:]}
    print("\nMann-Whitney p (sims to robust spec, censored):")
    for k, x in sorted(p.items(), key=lambda kv: kv[1]):
        print(f"  {k}: p={x:.4f}")
    Path("results/robust_stats.json").write_text(json.dumps({"summary": s, "pvalues": p}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
