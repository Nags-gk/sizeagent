"""Aggregate results/ into docs/data/*.json for the GitHub Pages dashboard and
render static PNG figures for the README."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sizeagent.circuit import space_size  # noqa: E402
from sizeagent.specs import Spec  # noqa: E402

RES, DOCS = ROOT / "results", ROOT / "docs"
LABEL = {"random": "Random search", "ga": "Genetic algorithm", "sa": "Simulated annealing",
         "tpe": "Bayesian opt. (TPE)", "surrogate_ga": "Surrogate-assisted GA", "agent": "LLM agent"}
COLOR = {"random": "#8a8f98", "ga": "#2a78d6", "sa": "#d97a1e", "tpe": "#7b53c4", "surrogate_ga": "#1a9e6e",
         "agent": "#c2405a"}


def _censored_median(xs):
    v = sorted(x if x else float("inf") for x in xs)
    m = statistics.median(v)
    return None if m == float("inf") else m


def bench_summary():
    rows = [json.loads(l) for l in (RES / "benchmark.jsonl").read_text().splitlines()]
    by = defaultdict(list)
    for r in rows:
        by[r["algo"]].append(r)
    for f in sorted(RES.glob("agent_*.json")):
        a = json.loads(f.read_text())
        curve, best = [], 1e9
        for t in a["trace"]:
            best = min(best, t["cost"])
            curve.append(best)
        by["agent"].append({"algo": "agent", "seed": f.stem, "budget": a["budget"], "sims": a["sims_used"],
                            "first_feasible": a["first_feasible_sim"], "best_cost": a["best"]["cost"],
                            "best_metrics": a["best"]["metrics"], "curve": curve})
    out = {}
    for algo, rs in by.items():
        ff = [r["first_feasible"] for r in rs if r["first_feasible"]]
        budget = max(r["budget"] for r in rs)
        feas_power = [r["best_metrics"]["power_uw"] for r in rs if r["first_feasible"]]
        L = budget
        curves = np.array([(r["curve"] + [r["curve"][-1]] * (L - len(r["curve"])))[:L] for r in rs])
        out[algo] = {"label": LABEL.get(algo, algo), "color": COLOR.get(algo, "#444"), "runs": len(rs),
                     "budget": budget, "success": len(ff),
                     # Median over ALL runs, counting a failed run as never reaching spec.
                     # None means fewer than half the runs met spec within the budget.
                     "median_first_feasible": _censored_median([r["first_feasible"] for r in rs]),
                     "median_first_feasible_successful": statistics.median(ff) if ff else None,
                     "first_feasible": [r["first_feasible"] for r in rs],
                     "median_best_power_uw": statistics.median(feas_power) if feas_power else None,
                     "min_best_power_uw": min(feas_power) if feas_power else None,
                     "median_curve": np.median(curves, axis=0).round(4).tolist(),
                     "q25_curve": np.percentile(curves, 25, axis=0).round(4).tolist(),
                     "q75_curve": np.percentile(curves, 75, axis=0).round(4).tolist()}
    return out


def plot_bench(b):
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, s in sorted(b.items(), key=lambda kv: kv[0] != "surrogate_ga"):
        x = np.arange(1, len(s["median_curve"]) + 1)
        ax.plot(x, s["median_curve"], color=s["color"], label=f'{s["label"]} ({s["success"]}/{s["runs"]} met spec)', lw=2)
        ax.fill_between(x, s["q25_curve"], s["q75_curve"], color=s["color"], alpha=0.12)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.text(3, 1.05, "below 1.0 = meets every spec", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("SPICE simulations")
    ax.set_ylabel("best cost so far (median, IQR)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "img" / "benchmark.png", dpi=150)


def plot_surrogate(s):
    ms = s["metrics"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    for ax, (k, m) in zip(axes.flat, ms.items()):
        ax.scatter(m["true"], m["pred"], s=5, alpha=0.5, color="#2a78d6")
        lo, hi = min(m["true"] + m["pred"]), max(m["true"] + m["pred"])
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8)
        ax.set_title(f'{m["label"]}\nR² = {m["r2"]:.3f}', fontsize=9)
        ax.tick_params(labelsize=7)
    fig.supxlabel("SPICE (held-out)", fontsize=9)
    fig.supylabel("Surrogate prediction", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "img" / "surrogate_parity.png", dpi=150)


def main():
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "img").mkdir(parents=True, exist_ok=True)
    spec = Spec()
    meta = {"spec": spec.__dict__, "spec_text": spec.describe(), "space_size": space_size()}
    b = bench_summary()
    plot_bench(b)
    data = {"meta": meta, "benchmark": b}
    if (RES / "surrogate_study.json").exists():
        s = json.loads((RES / "surrogate_study.json").read_text())
        plot_surrogate(s)
        data["surrogate"] = s
    if (RES / "pvt_study.json").exists():
        data["pvt"] = json.loads((RES / "pvt_study.json").read_text())
    agents = []
    for f in sorted(RES.glob("agent_*.json")):
        a = json.loads(f.read_text())
        agents.append({k: a[k] for k in ("provider", "model", "budget", "sims_used", "first_feasible_sim",
                                         "submitted", "rationale", "turns", "wall_s", "log", "best")})
    data["agents"] = agents
    if (RES / "surrogate_model.json").exists():
        data["model"] = json.loads((RES / "surrogate_model.json").read_text())
    (DOCS / "data" / "results.json").write_text(json.dumps(data, default=str))
    for algo, s in sorted(b.items()):
        print(f'{s["label"]:>24}: {s["success"]}/{s["runs"]} met spec, median sims-to-spec '
              f'{s["median_first_feasible"]}, median best power {s["median_best_power_uw"]}')


if __name__ == "__main__":
    main()
