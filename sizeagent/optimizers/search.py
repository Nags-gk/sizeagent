"""Baseline combinatorial optimizers: random search, genetic algorithm,
simulated annealing, and Bayesian optimization (TPE)."""
from __future__ import annotations

import math
import random

from ..specs import BudgetExceeded, Evaluator
from . import cardinality, decode, mutate_gene, random_vector, run_safely


def _f(ev: Evaluator, v: list[int]) -> float:
    return ev(decode(v))[0]


@run_safely
def random_search(ev: Evaluator, seed: int = 0) -> None:
    rng = random.Random(seed)
    while True:
        _f(ev, random_vector(rng))


@run_safely
def genetic_algorithm(ev: Evaluator, seed: int = 0, pop: int = 20, elite: int = 2,
                      tour: int = 3, pmut: float = 0.15) -> None:
    """Steady generational GA: tournament selection, uniform crossover,
    neighborhood mutation, elitism."""
    rng = random.Random(seed)
    card = cardinality()
    population = [random_vector(rng) for _ in range(pop)]
    scored = [(_f(ev, v), v) for v in population]
    while True:
        scored.sort(key=lambda t: t[0])
        nxt = [v for _, v in scored[:elite]]
        while len(nxt) < pop:
            a = min(rng.sample(scored, tour), key=lambda t: t[0])[1]
            b = min(rng.sample(scored, tour), key=lambda t: t[0])[1]
            child = [x if rng.random() < 0.5 else y for x, y in zip(a, b)]
            for i in range(len(child)):
                if rng.random() < pmut:
                    mutate_gene(child, i, rng, card)
            nxt.append(child)
        scored = [(_f(ev, v), v) for v in nxt]


@run_safely
def simulated_annealing(ev: Evaluator, seed: int = 0, t0: float = 1.0, t_end: float = 0.01,
                        restarts_after: int = 60) -> None:
    """Single-gene neighborhood moves with a geometric cooling schedule tied to the budget."""
    rng = random.Random(seed)
    card = cardinality()
    cur = random_vector(rng)
    c_cur = _f(ev, cur)
    best_c, stall = c_cur, 0
    while True:
        frac = min(ev.sims / ev.budget, 1.0)
        temp = t0 * (t_end / t0) ** frac
        cand = list(cur)
        for _ in range(rng.choice([1, 1, 2])):
            mutate_gene(cand, rng.randrange(len(cand)), rng, card)
        c = _f(ev, cand)
        if c < c_cur or rng.random() < math.exp(-(c - c_cur) / max(temp, 1e-9)):
            cur, c_cur = cand, c
        if c < best_c - 1e-9:
            best_c, stall = c, 0
        else:
            stall += 1
        if stall > restarts_after:          # restart from a random point
            cur, stall = random_vector(rng), 0
            c_cur = _f(ev, cur)


@run_safely
def tpe(ev: Evaluator, seed: int = 0) -> None:
    """Bayesian optimization with Optuna's Tree-structured Parzen Estimator over categorical genes."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    card = cardinality()
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=20, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        v = [trial.suggest_int(f"g{i}", 0, n - 1) for i, n in enumerate(card)]
        try:
            return _f(ev, v)
        except BudgetExceeded:      # stop cleanly instead of letting optuna log a failed trial
            study.stop()
            return float("inf")

    # Duplicate suggestions are free (cached), so allow many more trials than simulations.
    study.optimize(objective, n_trials=10 * ev.budget)


def _surrogate_ga(ev, seed=0, **kw):
    from ..surrogate import surrogate_ga
    return surrogate_ga(ev, seed, **kw)


ALGORITHMS = {
    "random": random_search,
    "ga": genetic_algorithm,
    "sa": simulated_annealing,
    "tpe": tpe,
    "surrogate_ga": _surrogate_ga,
}


def local_search(ev: Evaluator, start: list[int], sims: int, seed: int = 0) -> tuple[float, list[int]]:
    """Short annealing run seeded at `start`, limited to `sims` new simulations.
    Used by the LLM agent as a refinement tool."""
    from ..specs import BudgetExceeded
    rng = random.Random(seed)
    card = cardinality()
    stop_at = ev.sims + sims
    cur, c_cur = list(start), float("inf")
    best, c_best = list(cur), c_cur
    try:
        c_cur = c_best = _f(ev, start)       # may raise if the budget is already spent
        step = 0
        while ev.sims < stop_at and step < 20 * max(sims, 1):
            temp = 0.3 * (0.01 / 0.3) ** (step / max(sims, 1))
            cand = list(cur)
            for _ in range(rng.choice([1, 1, 2])):
                mutate_gene(cand, rng.randrange(len(cand)), rng, card)
            c = _f(ev, cand)
            if c < c_cur or rng.random() < math.exp(-(c - c_cur) / temp):
                cur, c_cur = cand, c
            if c < c_best:
                best, c_best = list(cand), c
            step += 1
    except BudgetExceeded:
        pass
    return c_best, best
