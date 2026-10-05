"""Does the surrogate's uncertainty help the optimizer? Ablate the exploration bonus.

Variants of the surrogate-assisted GA, each over the same seeds and 300-simulation budget as the
main benchmark (whose `surrogate_ga` runs are the uncalibrated, explore=0.5 baseline):
  sga_no_explore : explore=0 (mean prediction only, uncertainty ignored)
  sga_cal        : calibrated uncertainty, explore=0.5
  sga_cal_x1     : calibrated uncertainty, explore=1.0
Writes results/ablation.jsonl (same record format as benchmark.jsonl) and prints the comparison.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sizeagent.specs import Evaluator  # noqa: E402
from sizeagent.stats import mann_whitney_p, summarize  # noqa: E402
from sizeagent.surrogate import surrogate_ga  # noqa: E402

VARIANTS = {"sga_no_explore": {"explore": 0.0}, "sga_cal": {"calibrate": True},
            "sga_cal_x1": {"calibrate": True, "explore": 1.0}}


def one(job):
    name, seed, budget = job
    ev = Evaluator(budget=budget)
    t = time.time()
    surrogate_ga(ev, seed=seed, **VARIANTS[name])
    b = ev.best()
    return {"algo": name, "seed": seed, "budget": budget, "sims": ev.sims, "first_feasible": ev.first_feasible(),
            "best_cost": b["cost"], "best_metrics": b["metrics"], "best_design": b["design"],
            "curve": [r["best_cost"] for r in ev.trace], "wall_s": round(time.time() - t, 1)}


def report(out: Path, bench: Path, budget: int):
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    rows += [r for r in map(json.loads, bench.read_text().splitlines()) if r["algo"] == "surrogate_ga"]
    s = summarize(rows, budget)
    print("| Variant | Met spec | Median sims to spec (95% CI) |\n|---|---|---|")
    for a, v in s.items():
        lo, hi = v["median_sims_ci"]
        print(f"| {a} | {v['success']}/{v['n']} | {v['median_sims']:.0f} ({lo:.0f}-{hi:.0f}) |")
    for a in s:
        if a != "surrogate_ga":
            print(f"  {a} vs surrogate_ga: p={mann_whitney_p(s[a]['sims'], s['surrogate_ga']['sims']):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/ablation.jsonl")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.report_only:
        done = {(r["algo"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
        jobs = [(v, s, a.budget) for s in range(a.seeds) for v in a.variants.split(",") if (v, s) not in done]
        out.parent.mkdir(parents=True, exist_ok=True)
        with Pool(a.workers) as p, open(out, "a") as f:
            for r in p.imap_unordered(one, jobs):
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f'{r["algo"]:>15} seed={r["seed"]} first_feasible={r["first_feasible"]} ({r["wall_s"]}s)', flush=True)
    report(out, Path("results/benchmark.jsonl"), a.budget)


if __name__ == "__main__":
    main()
