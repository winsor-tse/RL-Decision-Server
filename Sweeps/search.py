"""Seeded designs using only the standard library and NumPy.

Numeric dimensions are uniform in either physical or log space. LHS stratifies
each latent dimension; integer/categorical conversion can produce duplicates.
"""

from itertools import product
import math
from numbers import Real

import numpy as np


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _choices(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("values must be a nonempty list")
    for value in values:
        if value is not None and not isinstance(value, (str, bool, Real)):
            raise ValueError("choices must be scalar JSON values")
        if isinstance(value, Real) and not math.isfinite(value):
            raise ValueError("choices must be finite")
    if len(set(values)) != len(values):
        raise ValueError("choices must be distinct")
    return list(values)


def _dimensions(space):
    if not isinstance(space, dict):
        raise ValueError("search space must be a dictionary")
    if any(not isinstance(name, str) or not name for name in space):
        raise ValueError("parameter names must be nonempty strings")
    return list(space)


def _validate_dimension(spec):
    if isinstance(spec, (list, tuple)):
        return {"values": _choices(spec)}
    if not isinstance(spec, dict):
        raise ValueError("dimension must be a list or a bounds dictionary")
    if "values" in spec:
        if set(spec) != {"values"}:
            raise ValueError("values cannot be combined with bounds")
        return {"values": _choices(spec["values"])}
    if set(spec) - {"min", "max", "scale", "type"}:
        raise ValueError(f"unknown dimension fields: {set(spec) - {'min', 'max', 'scale', 'type'}}")
    low, high = spec.get("min"), spec.get("max")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           for v in (low, high)) or low >= high:
        raise ValueError("bounds must be finite numbers with min < max")
    scale, kind = spec.get("scale", "linear"), spec.get("type", "float")
    if scale not in ("linear", "log") or kind not in ("float", "int"):
        raise ValueError("scale must be linear/log; type must be float/int")
    if scale == "log" and low <= 0:
        raise ValueError("log bounds must be positive")
    if kind == "int" and (int(low) != low or int(high) != high):
        raise ValueError("integer bounds must be integers")
    return {"min": low, "max": high, "scale": scale, "type": kind}


def _decode(spec, unit):
    if "values" in spec:
        return spec["values"][min(int(unit * len(spec["values"])), len(spec["values"]) - 1)]
    low, high = spec["min"], spec["max"]
    if spec["scale"] == "log":
        value = math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
    elif spec["type"] == "int":
        return min(int(high), int(low) + int(unit * (high - low + 1)))
    else:
        value = low + unit * (high - low)
    return max(int(low), min(int(high), round(value))) if spec["type"] == "int" else float(value)


def grid_search(space):
    """Iterate the full Cartesian product lazily, preserving parameter order.

    Each dimension must contain explicit values. An empty space yields one
    empty configuration, useful for a baseline repeated across seeds.
    """
    names = _dimensions(space)
    dimensions = [_validate_dimension(space[name]) for name in names]
    if any("values" not in spec for spec in dimensions):
        raise ValueError("grid dimensions require explicit values")
    return (dict(zip(names, values)) for values in product(*(d["values"] for d in dimensions)))


def _sample(space, samples, seed, stratified):
    samples = positive_integer(samples, "samples")
    names = _dimensions(space)
    dimensions = [_validate_dimension(space[name]) for name in names]
    rng = np.random.default_rng(seed)
    unit = np.empty((samples, len(names)))
    for column in range(len(names)):
        if stratified:
            unit[:, column] = (rng.permutation(samples) + rng.random(samples)) / samples
        else:
            unit[:, column] = rng.random(samples)
    return [
        {name: _decode(spec, row[j]) for j, (name, spec) in enumerate(zip(names, dimensions))}
        for row in unit
    ]


def latin_hypercube(space, samples, seed=0):
    """One jittered point per equal-probability stratum in each dimension."""
    return _sample(space, samples, seed, stratified=True)


def monte_carlo(space, samples, seed=0):
    """Independent draws; also usable as random hyperparameter search."""
    return _sample(space, samples, seed, stratified=False)
