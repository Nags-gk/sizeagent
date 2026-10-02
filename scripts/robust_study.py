"""Corner-aware vs nominal-only optimization.

Each algorithm runs with RobustEvaluator (nominal first, then 6 extreme PVT
corners for nominally feasible designs, all charged to the budget). The best
design of each run is then verified on the full 5 corners x 3 temperatures grid.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sizeagent.agent.tools import design_from_args  # noqa: E402
from sizeagent.optimizers.search import ALGORITHMS  # noqa: E402
from sizeagent.pvt import corner_sweep  # noqa: E402
from sizeagent.robust import RobustEvaluator  # noqa: E402
from sizeagent.specs import Spec  # noqa: E402


def one(job):
    algo, seed, budget = job
    ev = RobustEvaluator(budget=budget)
    t = time.time()
    ALGORITHMS[algo](ev, seed=seed)
    b = ev.best()
    full = corner_sweep(design_from_args(b["design"]), Spec(), workers=1)
    return {"algo": algo, "seed": seed, "budget": budget, "sims": ev.sims,
            "first_robust_feasible": ev.first_feasible(),
            "first_nominal_feasible": next((r["sim"] for r in ev.trace if r["nominal_feasible"]), None),
            "best_cost": b["cost"], "best_metrics": b["metrics"], "best_design": b["design"],
            "full_pass": full["pass_count"], "full_total": full["total"], "full_worst": full["worst"],
            "curve": [r["best_cost"] for r in ev.trace], "wall_s": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default="ga,surrogate_ga,sa")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/robust.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    done = {(r["algo"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    jobs = [(al, s, a.budget) for s in range(a.seeds) for al in a.algos.split(",") if (al, s) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as p, open(out, "a") as f:
        for r in p.imap_unordered(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f'{r["algo"]:>13} seed={r["seed"]} robust_feasible_at={r["first_robust_feasible"]} '
                  f'full_grid={r["full_pass"]}/{r["full_total"]} ({r["wall_s"]}s)', flush=True)


if __name__ == "__main__":
    main()
