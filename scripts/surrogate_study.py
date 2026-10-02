"""Surrogate-vs-SPICE correlation study.

Builds a dataset of SPICE-simulated designs (uniform random samples plus local
perturbations around good designs found by the benchmark), trains the MLP
ensemble on 80%, and reports held-out accuracy per metric.
"""
import argparse, json, random, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sizeagent.circuit import random_design  # noqa: E402
from sizeagent.optimizers import cardinality, decode, encode, mutate_gene  # noqa: E402
from sizeagent.spice import simulate  # noqa: E402
from sizeagent.surrogate import TARGETS, Surrogate, features, targets  # noqa: E402

METRIC_NAMES = {"gain_db": "DC gain (dB)", "log_ugbw": "log10 UGBW (MHz)", "pm_deg": "Phase margin (deg)",
                "log_power": "log10 power (uW)", "sat_margin_v": "Min sat. margin (V)", "vout_err_v": "|Vout - Vcm| (V)"}


def sim(vec):
    r = simulate(decode(vec))
    return vec, (r.metrics() if r.ok else None)


def build_dataset(n_random: int, n_local: int, seeds_from: Path, seed: int = 0):
    rng = random.Random(seed)
    vecs = [encode(random_design(rng)) for _ in range(n_random)]
    anchors = []
    if seeds_from.exists():
        for line in seeds_from.read_text().splitlines():
            r = json.loads(line)
            if r["first_feasible"]:
                from sizeagent.agent.tools import design_from_args
                anchors.append(encode(design_from_args(r["best_design"])))
    card = cardinality()
    for i in range(n_local if anchors else 0):
        v = list(rng.choice(anchors))
        for _ in range(rng.randint(1, 4)):
            mutate_gene(v, rng.randrange(len(v)), rng, card)
        vecs.append(v)
    with Pool(2) as p:
        rows = p.map(sim, vecs, chunksize=8)
    return [(v, m) for v, m in rows if m is not None], len(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=1200)
    ap.add_argument("--n-local", type=int, default=800)
    ap.add_argument("--out", default="results/surrogate_study.json")
    a = ap.parse_args()
    data, attempted = build_dataset(a.n_random, a.n_local, Path("results/benchmark.jsonl"))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(data))
    cut = int(0.8 * len(data))
    X = np.array([features(decode(v)) for v, _ in data])
    Y = np.array([targets(m) for _, m in data])
    tr, te = idx[:cut], idx[cut:]
    sur = Surrogate(n_models=4, epochs=300, seed=0).fit(X[tr], Y[tr])
    mu, sd = sur.predict(X[te])
    report = {"n_attempted": attempted, "n_converged": len(data), "n_train": len(tr), "n_test": len(te), "metrics": {}}
    for j, t in enumerate(TARGETS):
        y, p = Y[te, j], mu[:, j]
        ss_res, ss_tot = ((y - p) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        report["metrics"][t] = {"label": METRIC_NAMES[t], "r2": float(1 - ss_res / ss_tot),
                                "mae": float(np.abs(y - p).mean()),
                                "pearson": float(np.corrcoef(y, p)[0, 1]),
                                "true": y.round(4).tolist(), "pred": p.round(4).tolist()}
        print(f"{t:>13}: R2={report['metrics'][t]['r2']:.3f}  MAE={report['metrics'][t]['mae']:.4f}")
    Path(a.out).write_text(json.dumps(report))
    export_model(Surrogate(n_models=3, epochs=300, seed=1).fit(X, Y), "results/surrogate_model.json")


def export_model(sur, path):
    """Write ensemble weights + normalization for the in-browser predictor."""
    from sizeagent.circuit import CC_PF, GROUPS, IB_UA, MULT_CHOICES, RZ_KOHM, valid_geometries
    nets = []
    for net in sur.models:
        layers = [m for m in net if hasattr(m, "weight")]
        nets.append([{"W": l.weight.detach().numpy().round(5).tolist(), "b": l.bias.detach().numpy().round(5).tolist()}
                     for l in layers])
    Path(path).write_text(json.dumps({
        "nets": nets, "xm": sur.xm.tolist(), "xs": sur.xs.tolist(), "ym": sur.ym.tolist(), "ys": sur.ys.tolist(),
        "targets": TARGETS, "groups": GROUPS, "geometries": valid_geometries(), "mult": MULT_CHOICES,
        "cc_pf": CC_PF, "rz_kohm": RZ_KOHM, "ib_ua": IB_UA}))


if __name__ == "__main__":
    main()
