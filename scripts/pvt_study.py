"""PVT corner sweep and Monte Carlo mismatch for the best design of each optimizer."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sizeagent.agent.tools import design_from_args  # noqa: E402
from sizeagent.pvt import corner_sweep, monte_carlo  # noqa: E402
from sizeagent.specs import Spec  # noqa: E402


def main(n_mc: int = 100):
    rows = [json.loads(l) for l in Path("results/benchmark.jsonl").read_text().splitlines()]
    best = {}
    for r in rows:
        if r["first_feasible"] and (r["algo"] not in best or r["best_cost"] < best[r["algo"]]["best_cost"]):
            best[r["algo"]] = r
    spec = Spec()
    out = {}
    for algo, r in sorted(best.items()):
        d = design_from_args(r["best_design"])
        cs = corner_sweep(d, spec)
        mc = monte_carlo(d, n_mc)
        out[algo] = {"design": r["best_design"], "nominal": r["best_metrics"], "corners": cs, "mc": mc}
        print(f"{algo:>13}: corners pass {cs['pass_count']}/{cs['total']}, worst PM {cs['worst']['pm_deg']:.1f}, "
              f"offset sigma {mc['offset_sigma_mv']:.2f} mV")
    Path("results/pvt_study.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
