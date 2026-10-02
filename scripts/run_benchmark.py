"""Benchmark combinatorial optimizers on the SKY130 op-amp sizing task.

Each (algorithm, seed) run gets the same SPICE budget. Reports simulations to
first spec-meeting design and the best feasible power found.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sizeagent.optimizers.search import ALGORITHMS  # noqa: E402
from sizeagent.specs import Evaluator  # noqa: E402


def one(job):
    algo, seed, budget = job
    ev = Evaluator(budget=budget)
    t = time.time()
    ALGORITHMS[algo](ev, seed=seed)
    b = ev.best()
    return {"algo": algo, "seed": seed, "budget": budget, "sims": ev.sims,
            "first_feasible": ev.first_feasible(), "best_cost": b["cost"], "best_metrics": b["metrics"],
            "best_design": b["design"], "curve": [r["best_cost"] for r in ev.trace],
            "wall_s": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default=",".join(ALGORITHMS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/benchmark.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r["algo"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())}
    jobs = [(al, s, a.budget) for s in range(a.seeds) for al in a.algos.split(",") if (al, s) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as p, open(out, "a") as f:
        for r in p.imap_unordered(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f'{r["algo"]:>13} seed={r["seed"]} first_feasible={r["first_feasible"]} '
                  f'best_cost={r["best_cost"]:.3f} ({r["wall_s"]}s)', flush=True)


if __name__ == "__main__":
    main()
