"""Summarize results/benchmark.jsonl with confidence intervals and pairwise tests.

Writes results/stats.json and prints a markdown table.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sizeagent.stats import pairwise_pvalues, summarize  # noqa: E402


def main(path: str = "results/benchmark.jsonl") -> None:
    runs = [json.loads(line) for line in Path(path).read_text().splitlines()]
    budget = runs[0]["budget"]
    s = summarize(runs, budget)
    p = pairwise_pvalues(s)
    print("| Algorithm | Success (95% CI) | Median sims to spec (95% CI) |\n|---|---|---|")
    for a, v in s.items():
        lo, hi = v["success_ci"]
        mlo, mhi = v["median_sims_ci"]
        print(f"| {a} | {v['success']}/{v['n']} ({lo:.0%}-{hi:.0%}) | {v['median_sims']:.0f} ({mlo:.0f}-{mhi:.0f}) |")
    print("\nMann-Whitney p (sims to spec; failures censored at budget+1):")
    for k, x in sorted(p.items(), key=lambda kv: kv[1]):
        print(f"  {k}: p={x:.4f}")
    Path("results/stats.json").write_text(json.dumps({"summary": s, "pvalues": p}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
