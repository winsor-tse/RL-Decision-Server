import os
import sys
import time
import torch
import numpy as np
import minari

# Ensure repo root is on PYTHONPATH so Custom_enviornments can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Offline.any_percent_bc import Actor, qlearning_dataset, best_trajectories_ids, compute_mean_std, normalize_states, wrap_env, evaluate

# Config
CHECKPOINT_DIR = os.path.join(repo_root, 'runs', 'bc-BC-v0-fa01204e')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'final_checkpoint.pt')
DATASET_ID = r"C:\Users\Cliff\.minari\datasets\env16\BC-v0"
NUM_EPISODES = 5
DEVICE = 'cpu'

print('Loading checkpoint:', CHECKPOINT_PATH)
state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Load dataset and prepare qdataset (use top_fraction=1.0 to include all)
dataset = minari.load_dataset(DATASET_ID)
traj_ids = best_trajectories_ids(dataset, top_fraction=1.0, gamma=0.99)
qdataset = qlearning_dataset(dataset, traj_ids=traj_ids)

# Compute normalization
state_mean, state_std = compute_mean_std(qdataset['observations'], eps=1e-3)
qdataset['observations'] = normalize_states(qdataset['observations'], state_mean, state_std)
qdataset['next_observations'] = normalize_states(qdataset['next_observations'], state_mean, state_std)

# Derive dims
state_dim = int(qdataset['observations'].shape[1])
action_dim = 1
max_action = float(max(1.0, np.max(np.abs(qdataset['actions']))))

# Build actor and load weights
actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
actor.load_state_dict(state['actor'])

# Recover environment for live evaluation
print('Recovering environment from dataset (this will try to import and construct the custom env)...')
eval_env = dataset.recover_environment()
# If the recovered env is the BC variant (which enforces matching captured actions),
# construct the live Env16 instead so the policy can act independently.
try:
    env_cls = None
    # Try to detect Env16BC by name
    if env_cls is None and eval_env.__class__.__name__ == 'Env16BC':
        from Custom_enviornments.Test_Env.Env_16 import Env16
        print('Recovered Env16BC; constructing Env16 for live policy evaluation')
        eval_env = Env16(config=eval_env.config)

    print('Wrapping environment with normalization')
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    # Run evaluation episodes
    print(f'Starting evaluation for {NUM_EPISODES} episodes. Make sure the game client/server is running and connected.')
    results = evaluate(eval_env, actor, num_episodes=NUM_EPISODES, seed=0, device=DEVICE)
except Exception as e:
    print('Live evaluation failed during env construction or rollout:', e)
    raise

print('Episode returns:', results)
print('Mean return:', results.mean())

# Close environment
try:
    eval_env.close()
except Exception:
    pass
