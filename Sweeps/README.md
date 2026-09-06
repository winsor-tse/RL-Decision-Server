# Local sweeps

Grid search, Latin hypercube sampling (LHS), Monte Carlo/random search, and
uncertainty estimates implemented here with the standard library and NumPy.
No Ray, Dakota, Optuna, SciPy, sklearn, PufferLib, or GPyTorch is needed for
these methods. Live trials use the existing repo dependencies and trainers.

`PROTEIN.py` is the separately copied PufferAI implementation. It is **not**
imported or selected by these tools and still has its original external
imports (including PufferLib, GPyTorch, SciPy, and sklearn). Vendoring that file
does not remove its dependencies. This implementation does not replace its
Gaussian processes or claim to implement Protein. Upstream information:
[Protein](https://puffer.ai/blog.html),
[source](https://github.com/PufferAI/PufferLib/blob/master/pufferlib/sweep.py).

## Preview a design

Run from the repository root with the repo virtual environment active:

```powershell
python -m Sweeps plan Sweeps/examples/grid.json --output runs/grid-plan.json
python -m Sweeps plan Sweeps/examples/lhs.json --output runs/lhs-plan.json
```

Planning does not connect to the game, import Protein, or start training. The
grid example has 27 configurations x 3 seeds = **81 live training runs**, each
followed by 20 evaluation episodes. The LHS example has 12 configurations x 3
seeds = **36 runs**. Edit the JSON to choose a smaller or larger study.
These are example ranges, not measured optimal settings.

`max_trials` defaults to 100,000 and rejects larger plans before materializing
them; raise it explicitly for larger studies. The `grid_search` Python API
itself yields configurations lazily. Every grid dimension has explicit values.

LHS/random dimensions accept:

```json
{
  "learning_rate": {"min": 0.00005, "max": 0.001, "scale": "log"},
  "update_epochs": {"min": 2, "max": 8, "type": "int"},
  "num_steps": {"values": [128, 256, 512]}
}
```

The default scale is `linear` and type is `float`. Bounds are inclusive for
integers. Linear integers and categorical choices have equal probability;
log integers are sampled in log space and rounded. Zero cannot be a log bound.
Parameters are independent. LHS stratification is in latent probability space;
rounding/categorical mapping can create duplicate configurations. The plan
removes these duplicates and reports their count, so the resulting discrete
design can have fewer points and no longer has an exact Latin property.

Choose `"method": "monte_carlo"` (or `"random"`) with the LHS schema for
independent sampling. `design_seed` controls sampling and trial order;
`training_seeds` controls repeated training runs. Trial order is shuffled to
reduce confounding with time. Bounds do not automatically represent a physical
uncertainty distribution: choose them according to your experimental question.

## Execute sequentially

Start the external game/extension as usual, then:

```powershell
python -m Sweeps run Sweeps/examples/lhs.json --output runs/sweeps/lhs-study
python -m Sweeps report runs/sweeps/lhs-study --output runs/sweeps/lhs-study/summary.json
```

`run` takes the design config, not the generated plan JSON. It rebuilds and
saves the same deterministic plan to the output directory. Repeating the same
command resumes the study by skipping completed results. A different plan in
that directory is rejected. Keep code, environment/reward rules, bridge
configuration, and game conditions fixed when resuming; these are not pinned
or automatically restored by the runner.

Both `ppo` and `ppo_lstm` are supported. The bridge comes from
`Automation/automation_config.yaml`; override with `--automation-config PATH`.
The automation training/inference selection and restore path are ignored.
Sweep trials always initialize fresh weights. TensorBoard can be started
separately with `python -m Automation.tensorboard_server --logdir runs`.

Each trial starts a bridge, trains, releases the trainer's ZMQ connection,
evaluates the exact saved checkpoint, and stops the bridge. Only one trial
runs at a time. The game must reconnect across bridge restarts. Each trial has:

- `model.pt`, `training.log`, and `evaluation.log`.
- `evaluation.json` containing episode returns, wins, evaluation seed, and
  deterministic/sampled policy mode.
- `result.json` containing parameters, training seed, actual training steps,
  training/evaluation/total seconds, and evaluation measurements.
- `status.json` recording a running, complete, or failed trial.

Trainer TensorBoard data remains under its normal `runs/sweep_*` directory.
The default total budget is 20,480 steps, divisible by the example rollout
lengths. Plans require complete rollouts rather than silently dropping steps.
Minibatches must divide the rollout and contain at least two samples.

`trial_timeout_seconds` defaults to six hours, including bridge readiness,
training, and evaluation; process startup/cleanup adds a small overhead.
Failure or interruption stops the sweep and cleans up its child processes.
On resume, incomplete trials restart from scratch; there is no partial PPO
training-state resume. Failures are not assigned a low policy score.

`runs/.sweep-live.lock` excludes other sweep runners in this checkout. It does
not control manually launched trainers/recorders or another checkout. Stop
those before running a sweep. After a force-killed runner, check the PID in
the lock and clean up its processes before manually removing that stale lock.

The runner uses the existing Env16 reset/step protocol. It does not add game
seeding, verified post-reset observations, or deterministic world resets.
Python seeds seed training/policy sampling only. Confirm consistent game
conditions before interpreting comparisons. Evaluation is deterministic by
default; set `"deterministic": false` for sampled actions.

## Uncertainty quantification

To measure a fixed configuration across seeds, use an empty `parameters`
dictionary and place its settings in `fixed`:

```powershell
python -m Sweeps run Sweeps/examples/uq.json --output runs/sweeps/baseline-uq
python -m Sweeps report runs/sweeps/baseline-uq --output runs/sweeps/baseline-uq/summary.json --confidence 0.95 --resamples 5000
```

For each configuration the report includes:

- Per-policy return means, sample SDs, and percentile bootstrap mean intervals.
- Per-policy win-rate Wilson intervals, including zero/all-win cases.
- Hierarchical bootstrap intervals that resample training runs, then episodes
  within those runs. Runs receive equal weight even with unequal episode counts.
- A bootstrap of per-training-run mean returns, with its sample SD exposing
  variation across trained policies.

Win rates and intervals are fractions between 0 and 1. `mean_ci: null` means
insufficient independent observations: a single training run cannot estimate
cross-training uncertainty. Partial reports identify completed/planned seed
counts and do not automatically rank incomplete configurations.

Bootstrap intervals with few runs can be unstable or degenerate. The methods
assume independent training runs and conditionally independent evaluation
episodes. They do not correct drift in the external game, persistent episode
correlations, or selection bias from choosing the best of many trials. Use
fresh final evaluations and repeated training seeds for finalists. UQ here is
not Sobol sensitivity analysis, polynomial chaos, or a Bayesian model of the
environment. LHS/Monte Carlo can also drive user-defined uncertainty propagation
through the Python API; no game-side scenario controls are fabricated.

## Python API

```python
from Sweeps import grid_search, latin_hypercube, monte_carlo, summarize_runs

design = latin_hypercube({
    "learning_rate": {"min": 5e-5, "max": 1e-3, "scale": "log"},
    "num_steps": [128, 256, 512],
}, samples=12, seed=42)

# Your own sequential experiment/scenario driver can consume these dictionaries.
for params in design:
    print(params)

summary = summarize_runs([
    {"seed": 1, "episode_returns": [10, 20, -5], "wins": 1},
    {"seed": 2, "episode_returns": [15, 25, 10], "wins": 2},
], seed=42)
```

The statistical implementations follow standard
[Latin hypercube stratification](https://tmap8.inl.gov/source/samplers/LatinHypercubeSampler.html)
and the [Wilson score interval](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
No dependency installation or live game access is needed to run their tests:

```powershell
python -m unittest Tests.test_sweeps
```
