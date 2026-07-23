# Improvement Roadmap

Ideas for extending training, fine-tuning, curriculum learning, replay storage, and behavior cloning.

## 1. Fine-Tune Models For Different Scenarios

Goal: take a trained model and continue training it on a different class/scenario env.

Steps:

1. Keep observation size and action size stable when possible.
2. Create a new env folder, for example:

```text
Custom_enviornments/
  Boss_Env/
    Env_16.py
    Env_conditions.py
```

3. Change scenario-specific rewards and end conditions in that env's `Env_conditions.py`.
4. Load the old DQN checkpoint into the new env's network.
5. Continue training with a smaller learning rate.
6. Use lower exploration than from-scratch training.
7. Save the fine-tuned model separately.

Example fine-tuning defaults:

```bash
--learning-rate 1e-4 --start-e 0.2 --end-e 0.02
```

## 3. Curriculum Training

Goal: train through increasingly difficult scenarios.

Example curriculum:

```text
stage_1: easy monsters
stage_2: stronger monsters
stage_3: multiple enemies
stage_4: boss fight
```

Steps:

1. Define several env folders or condition configs.
2. Train from scratch on stage 1.
3. Save checkpoint.
4. Load checkpoint into stage 2.
5. Continue training.
6. Repeat through harder stages.
7. Track metadata for each stage.

Metadata to store:

- source checkpoint
- target environment
- reward config
- total timesteps
- mean return
- date trained

Possible future config:

```yaml
curriculum:
  - env: Easy_Env
    timesteps: 5000
  - env: Normal_Env
    timesteps: 10000
  - env: Boss_Env
    timesteps: 15000
```

## 6. Behavior Cloning / Inverse RL

Goal: train a model from high-return state/action pairs.

Behavior cloning learns:

```text
state -> action
```

Steps:

1. Collect expert or high-return transitions.
2. Build a dataset:

```text
X = states
y = actions
```

3. Train the Q-network or policy head with classification loss.
4. Save the behavior-cloned model.
5. Use that model as initialization for DQN or even for critic in PPO.
6. Fine-tune with normal RL afterward.

Example pipeline:

```text
Train DQN -> collect elite data -> behavior clone -> fine-tune with DQN
```

This can help when rewards are sparse or early training is unstable.

## Recommended Build Order

1. Improve checkpoint format.
2. Add `Training/Load_DQN_server.py`.
3. Add fine-tuning support.
4. Add experience logging.
5. Add elite episode filtering.
6. Add behavior cloning.
7. Add curriculum runner.

Recommended future file layout:

```text
Training/
  DQN_server.py
  Load_DQN_server.py
  Curriculum_train.py

Inference/
  dqn_eval.py

Data/
  experience_store.py
  rl_experiences.sqlite

Imitation/
  behavior_clone.py

Utils/
  checkpointing.py
  buffers.py
```

## Important Constraint

Keep observation and action dimensions stable across models that should be resumed or fine-tuned. If those dimensions change, checkpoint loading needs partial weight loading, model surgery, or a conversion layer.


## Red Section

Two threads for actions, one for movement/facing and one for casting.