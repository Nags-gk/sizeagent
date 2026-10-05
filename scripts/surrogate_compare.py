"""Compare surrogates on one shared SPICE dataset, judged the way the optimizer uses them.

Models: the PyTorch MLP ensemble (current), Gaussian process, random forest, bagged boosting.
Per model and metric: held-out R^2/MAE; uncertainty calibration (90% coverage, sd-vs-error rank
correlation); and feasibility ranking (AUROC, precision@k of predicted-best designs).
Averaged over several random train/test splits. Writes results/surrogate_compare.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sizeagent.baselines import MODELS  # noqa: E402
from sizeagent.dataset import build_dataset, load_dataset, save_dataset  # noqa: E402
from sizeagent.optimizers import decode  # noqa: E402
from sizeagent.specs import Spec  # noqa: E402
from sizeagent.surrogate import TARGETS, Surrogate, features, targets  # noqa: E402
from sizeagent.surrogate_eval import calibration, feasibility_ranking, r2_mae  # noqa: E402

DATA = Path("results/surrogate_dataset.jsonl")


def make_models(fast: bool):
    ms = {"mlp_ensemble": lambda s: Surrogate(n_models=4, epochs=100 if fast else 300, seed=s)}
    ms.update({k: (lambda s, c=c: c(seed=s)) for k, c in MODELS.items()})
    return ms


def evaluate(X, Y, tr, te, model):
    t = time.time()
    mu, sd = model.fit(X[tr], Y[tr]).predict(X[te])
    fit_s = time.time() - t
    out = {"fit_predict_s": round(fit_s, 1), "targets": {}}
    for j, name in enumerate(TARGETS):
        r2, mae = r2_mae(Y[te, j], mu[:, j])
        out["targets"][name] = {"r2": r2, "mae": mae, **calibration(Y[te, j], mu[:, j], sd[:, j])}
    out["ranking"] = feasibility_ranking(Y[te], mu, Spec())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="simulate a fresh dataset (a few minutes)")
    ap.add_argument("--n-random", type=int, default=1200)
    ap.add_argument("--n-local", type=int, default=800)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="results/surrogate_compare.json")
    a = ap.parse_args()
    if a.build or not DATA.exists():
        data, attempted = build_dataset(a.n_random, a.n_local, Path("results/benchmark.jsonl"), workers=6)
        save_dataset(DATA, data)
        print(f"simulated {attempted}, converged {len(data)} -> {DATA}")
    data = load_dataset(DATA)
    X = np.array([features(decode(v)) for v, _ in data])
    Y = np.array([targets(m) for _, m in data])
    res: dict = {"n": len(data), "splits": a.splits, "models": {}}
    for name, mk in make_models(a.fast).items():
        runs = []
        for s in range(a.splits):
            perm = np.random.default_rng(s).permutation(len(data))
            cut = int(0.8 * len(data))
            runs.append(evaluate(X, Y, perm[:cut], perm[cut:], mk(s)))
            print(f"{name} split {s}: {runs[-1]['fit_predict_s']}s  PM R2={runs[-1]['targets']['pm_deg']['r2']:.3f}  "
                  f"AUROC={runs[-1]['ranking']['auroc']:.3f}", flush=True)
        res["models"][name] = {"runs": runs}
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
