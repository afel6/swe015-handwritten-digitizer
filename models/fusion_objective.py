"""Continuous dev objective and deterministic search for candidate-fusion weights.

Objective (fixed before tuning): for every dev item whose target word is in the candidate
pool, take the log-softmax of the fused candidate scores and read off the target's entry;
the objective is the mean of these values (higher is better, 0 is the maximum). Items whose
target is not in the pool cannot be influenced by the weights and are excluded, as are items
the numeric-literal guard decides independently of the weights.

Scale: the log-softmax objective always prefers larger weights once the target ranks first
(scaling every score up sharpens the softmax), which would push any free grid to its largest
values. The fused weights are therefore searched on a simplex (they sum to 1), so only their
ratios are tuned and the score scale is fixed; non-fused hyper-parameters (BERT's alpha) are
searched over a plain grid.

Tie-break (fixed before tuning): candidates are visited in ascending lexicographic order of
their values (parameters in the order given); a later candidate replaces the best only if its
objective is finite and strictly better. Ties therefore resolve to the lexicographically
smallest weights.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["grid_search", "mean_target_log_softmax", "search", "simplex_points"]


def mean_target_log_softmax(scored_items: Sequence[tuple[np.ndarray, int]]) -> float:
    """Mean log-softmax probability of the target among each item's candidate scores.

    scored_items holds (scores, target_index) pairs; only items with the target in the pool
    belong here. Returns -inf for an empty sequence.
    """
    if not scored_items:
        return -math.inf
    total = 0.0
    for scores, target_index in scored_items:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 1 or not 0 <= target_index < arr.size:
            raise ValueError("Each item needs a 1D score array and a target index inside it.")
        shifted = arr - arr.max()
        total += float(shifted[target_index] - math.log(float(np.exp(shifted).sum())))
    return total / len(scored_items)


def simplex_points(names: Sequence[str], step: float = 0.05, minimum: float = 0.05) -> list[dict[str, float]]:
    """All weight vectors on a step grid with every weight >= minimum and the weights summing to 1."""
    if not names or step <= 0 or minimum < 0 or minimum * len(names) > 1.0 + 1e-9:
        raise ValueError("simplex needs names, a positive step and a feasible minimum.")
    units = round(1.0 / step)
    low = round(minimum / step)
    points = []
    for combo in itertools.product(range(low, units + 1), repeat=len(names) - 1):
        last = units - sum(combo)
        if last >= low:
            points.append({n: round(u * step, 10) for n, u in zip(names, (*combo, last), strict=True)})
    return sorted(points, key=lambda p: tuple(p[n] for n in names))


def search(candidates: Iterable[Mapping[str, float]], objective: Callable[[dict[str, float]], float]) -> tuple[dict[str, float], float]:
    """Return the first candidate with the strictly best finite objective (see module docstring)."""
    best_params: dict[str, float] = {}
    best_value = -math.inf
    for cand in candidates:
        params = dict(cand)
        value = objective(params)
        if math.isfinite(value) and (value > best_value or not best_params):
            best_params, best_value = params, value
    if not best_params:
        raise ValueError("No candidate produced a finite objective.")
    return best_params, best_value


def grid_search(grid: Mapping[str, Sequence[float]], objective: Callable[[dict[str, float]], float]) -> tuple[dict[str, float], float]:
    """Plain grid search with the module's tie-break (for hyper-parameters that are not fusion weights)."""
    names = list(grid)
    if not names or any(not grid[n] for n in names):
        raise ValueError("grid must name at least one parameter, each with at least one value.")
    points = (dict(zip(names, values, strict=True)) for values in itertools.product(*(sorted(float(v) for v in grid[n]) for n in names)))
    return search(points, objective)
