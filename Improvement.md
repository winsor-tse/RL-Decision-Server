# Improvement Roadmap

Ideas for extending training, fine-tuning, curriculum learning, replay storage, and behavior cloning.

## 1. Resume DQN Training From Checkpoints

Goal: create `Training/Load_DQN_server.py` to load a previous run and continue training.

Steps:

1. Save richer checkpoints from `Training/DQN_server.py`.
2. Include model weights, target model weights, optimizer state, global step, and args.
3. Build the env before loading the model.
4. Create `QNetwork(env)` exactly like normal training.
5. Load the checkpoint weights into `q_network`.
6. Load the same weights into `target_network`.
7. Optionally load optimizer state so training resumes more smoothly.
8. Resume epsilon carefully so the model does not restart with fully random behavior.

Recommended checkpoint shape:

```python
{
    "model": q_network.state_dict(),
    "target_model": target_network.state_dict(),
    "optimizer": optimizer.state_dict(),
    "global_step": global_step,
    "args": vars(args),
}
```

Example command:

```bash
python Training/Load_DQN_server.py --checkpoint runs/.../DQN_server.pt
```

## 2. Fine-Tune Models For Different Scenarios

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

## 4. Store Top States And Actions

Goal: create a local experience database for high-return behavior.

Recommended first option: SQLite.

Why SQLite:

- local file
- no server required
- easy queries
- good enough for replay and behavior cloning experiments

Example path:

```text
Data/rl_experiences.sqlite
```

Recommended tables:

```text
experiences
episodes
models
```

Important `experiences` columns:

```text
id
episode_id
state
action
reward
next_state
done
step
episode_return
env_name
model_path
created_at
```

Start by storing `state` and `next_state` as JSON lists. Use binary or compressed NumPy storage later if needed.

## 5. Store Highest-Return Episodes

There are two good approaches.

Option A: store all transitions, query the best later.

Example query:

```sql
SELECT *
FROM experiences
WHERE episode_return > 50
ORDER BY reward DESC;
```

Option B: only save elite episodes.

Steps:

1. Track episode return during training.
2. Keep transitions for the current episode in memory.
3. If episode return is above a threshold, save the episode.
4. Keep the top N episodes per environment.

This creates cleaner data for behavior cloning.

## 6. Behavior Cloning

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
5. Use that model as initialization for DQN.
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