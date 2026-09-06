"""Bootstrap means and binomial intervals without SciPy or sklearn.

Training runs are the independent units for cross-seed uncertainty. Intervals
are descriptive for the evaluated settings, not a correction for selecting the
best of many configurations. Independence/stationarity must be checked by the
experimenter for an external game.
"""

from statistics import NormalDist

import numpy as np

from Sweeps.search import positive_integer


def _values(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be a nonempty finite one-dimensional sequence")
    return array


def _confidence(confidence):
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    return (1 - confidence) / 2


def bootstrap_mean(values, *, confidence=0.95, resamples=2000, seed=0):
    """Percentile bootstrap CI for a mean of independent observations.

    With fewer than two observations the interval and sample SD are undefined.
    """
    array = _values(values)
    alpha = _confidence(confidence)
    resamples = positive_integer(resamples, "resamples")
    interval = None
    if len(array) > 1:
        rng = np.random.default_rng(seed)
        means = np.array([rng.choice(array, len(array), replace=True).mean() for _ in range(resamples)])
        interval = np.quantile(means, [alpha, 1 - alpha]).tolist()
    return {
        "n": len(array), "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "confidence": confidence, "mean_ci": interval,
    }


def hierarchical_mean(runs, *, confidence=0.95, resamples=2000, seed=0):
    """Resample runs, then episodes within each selected run, with equal run weight.

    At least two independent training runs are required for an interval. This
    estimates both between-run and within-run evaluation variation.
    """
    arrays = [_values(run) for run in runs]
    if not arrays:
        raise ValueError("at least one run is required")
    alpha = _confidence(confidence)
    resamples = positive_integer(resamples, "resamples")
    interval = None
    if len(arrays) > 1:
        rng = np.random.default_rng(seed)
        means = []
        for _ in range(resamples):
            selected = rng.integers(len(arrays), size=len(arrays))
            means.append(np.mean([
                rng.choice(arrays[i], len(arrays[i]), replace=True).mean() for i in selected
            ]))
        interval = np.quantile(means, [alpha, 1 - alpha]).tolist()
    return {
        "runs": len(arrays), "episodes": sum(map(len, arrays)),
        "mean": float(np.mean([array.mean() for array in arrays])),
        "confidence": confidence, "mean_ci": interval,
    }


def wilson_interval(wins, episodes, *, confidence=0.95):
    """Wilson score interval for one fixed policy's independent binary outcomes."""
    episodes = positive_integer(episodes, "episodes")
    if isinstance(wins, bool) or not isinstance(wins, (int, np.integer)) or not 0 <= wins <= episodes:
        raise ValueError("wins must be an integer between zero and episodes")
    alpha = _confidence(confidence)
    z = NormalDist().inv_cdf(1 - alpha)
    rate = wins / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    radius = z / denominator * np.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes**2))
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]


def summarize_runs(runs, *, confidence=0.95, resamples=2000, seed=0):
    """Summarize results for ONE configuration, with unique training seeds.

    Each run supplies seed, episode_returns, and wins. Win rates are fractions.
    Do not pool episodes across trained policies into one binomial interval.
    """
    if not runs or len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("provide at least one run with distinct training seeds")
    settings = dict(confidence=confidence, resamples=resamples, seed=seed)
    per_run, returns, outcomes = [], [], []
    for run in runs:
        values = _values(run["episode_returns"])
        wins = run["wins"]
        interval = wilson_interval(wins, len(values), confidence=confidence)
        returns.append(values)
        outcomes.append([1.0] * wins + [0.0] * (len(values) - wins))
        per_run.append({
            "seed": run["seed"], "return": bootstrap_mean(values, **settings),
            "win_rate": wins / len(values), "win_rate_ci": interval,
        })
    return {
        "per_run": per_run,
        "return_across_runs": hierarchical_mean(returns, **settings),
        "win_rate_across_runs": hierarchical_mean(outcomes, **settings),
        "training_run_mean_return": bootstrap_mean([r.mean() for r in returns], **settings),
    }
