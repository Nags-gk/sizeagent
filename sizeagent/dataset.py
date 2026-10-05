"""SPICE-simulated design datasets: build once, save as JSONL, reuse across surrogate studies."""
from __future__ import annotations

import json
import random
from multiprocessing import Pool
from pathlib import Path

from .circuit import random_design
from .optimizers import cardinality, decode, encode, mutate_gene
from .spice import simulate


def _sim(vec: list[int]):
    r = simulate(decode(vec))
    return vec, (r.metrics() if r.ok else None)


def build_dataset(n_random: int, n_local: int, anchors_from: Path | None = None, seed: int = 0,
                  workers: int = 4) -> tuple[list[tuple[list[int], dict]], int]:
    """Uniform random designs plus local perturbations of feasible designs from a benchmark file.
    Returns (converged (vector, metrics) pairs, number attempted)."""
    from .agent.tools import design_from_args
    rng = random.Random(seed)
    vecs = [encode(random_design(rng)) for _ in range(n_random)]
    anchors = []
    if anchors_from is not None and anchors_from.exists():
        for line in anchors_from.read_text().splitlines():
            r = json.loads(line)
            if r.get("first_feasible"):
                anchors.append(encode(design_from_args(r["best_design"])))
    card = cardinality()
    for _ in range(n_local if anchors else 0):
        v = list(rng.choice(anchors))
        for _ in range(rng.randint(1, 4)):
            mutate_gene(v, rng.randrange(len(v)), rng, card)
        vecs.append(v)
    with Pool(workers) as p:
        rows = p.map(_sim, vecs, chunksize=8)
    return [(v, m) for v, m in rows if m is not None], len(vecs)


def save_dataset(path: Path, data: list[tuple[list[int], dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"vec": v, "metrics": m}) for v, m in data) + "\n")


def load_dataset(path: Path) -> list[tuple[list[int], dict]]:
    return [(r["vec"], r["metrics"]) for r in map(json.loads, path.read_text().splitlines())]
