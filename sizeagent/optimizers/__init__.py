"""Combinatorial sizing optimizers. Each takes an Evaluator and a seed and runs
until the evaluator's simulation budget is exhausted."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from functools import cache

from ..circuit import CC_PF, GROUPS, IB_UA, MULT_CHOICES, MULT_KEYS, RZ_KOHM, Design, random_design, valid_geometries
from ..specs import BudgetExceeded, Evaluator

# Genome: 4 geometry genes, 5 multiplier genes, Cc, Rz, Ibias (all integer indices).
GENES: list[tuple[str, str]] = ([("geo", g) for g in GROUPS] + [("mult", k) for k in MULT_KEYS]
                                 + [("cc", ""), ("rz", ""), ("ib", "")])
SCALAR_GRIDS: dict[str, Sequence[float]] = {"cc": CC_PF, "rz": RZ_KOHM, "ib": IB_UA}


def cardinality() -> list[int]:
    geos = valid_geometries()
    out = []
    for kind, name in GENES:
        if kind == "geo":
            out.append(len(geos[GROUPS[name]]))
        elif kind == "mult":
            out.append(len(MULT_CHOICES))
        else:
            out.append(len(SCALAR_GRIDS[kind]))
    return out


def encode(d: Design) -> list[int]:
    v = []
    for kind, name in GENES:
        if kind == "geo":
            v.append(d.geo[name])
        elif kind == "mult":
            v.append(MULT_CHOICES.index(d.mult[name]))
        else:
            v.append(SCALAR_GRIDS[kind].index(getattr(d, kind)))
    return v


def decode(v: list[int]) -> Design:
    geo, mult = {}, {}
    i = 0
    for kind, name in GENES:
        if kind == "geo":
            geo[name] = v[i]
        elif kind == "mult":
            mult[name] = MULT_CHOICES[v[i]]
        i += 1
    return Design(geo=geo, mult=mult, cc=CC_PF[v[9]], rz=RZ_KOHM[v[10]], ib=IB_UA[v[11]])


@cache
def geo_neighbors(dev: str, k: int = 6) -> list[list[int]]:
    """k nearest characterized geometries in log(W, L) space, per geometry."""
    pairs = valid_geometries()[dev]
    out = []
    for i, (w, l) in enumerate(pairs):
        dist = sorted((math.log(w / w2) ** 2 + math.log(l / l2) ** 2, j) for j, (w2, l2) in enumerate(pairs) if j != i)
        out.append([j for _, j in dist[:k]])
    return out


def mutate_gene(v: list[int], i: int, rng: random.Random, card: list[int]) -> None:
    kind, name = GENES[i]
    if kind == "geo":
        v[i] = rng.choice(geo_neighbors(GROUPS[name])[v[i]])
    else:
        step = rng.choice([-2, -1, 1, 2])
        v[i] = min(card[i] - 1, max(0, v[i] + step))


def random_vector(rng: random.Random) -> list[int]:
    return encode(random_design(rng))


def run_safely(fn):
    """Decorator: stop cleanly when the simulation budget runs out."""
    def wrapper(ev: Evaluator, seed: int = 0, **kw):
        try:
            fn(ev, seed, **kw)
        except BudgetExceeded:
            pass
        return ev
    wrapper.__name__ = fn.__name__
    return wrapper
