import os
import torch
import minari
import numpy as np
from any_percent_bc import Actor, qlearning_dataset, best_trajectories_ids, TrainConfig

CHECKPOINT_DIR = r"C:\Users\Cliff\experiments\bc\bc-BC-v0-fa01204e"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "final_checkpoint.pt")
DATASET_ID = r"C:\Users\Cliff\.minari\datasets\env16\BC-v0"

print('Loading checkpoint:', CHECKPOINT_PATH)
state = torch.load(CHECKPOINT_PATH, map_location='cpu')

# Load dataset and build qdataset using all trajectories
dataset = minari.load_dataset(DATASET_ID)
traj_ids = best_trajectories_ids(dataset, top_fraction=1.0, gamma=0.99)
qdataset = qlearning_dataset(dataset, traj_ids=traj_ids)

# Determine dims
state_dim = int(qdataset['observations'].shape[1])
action_dim = 1
max_action = float(max(1.0, np.max(np.abs(qdataset['actions']))))

# Build actor and load weights
actor = Actor(state_dim, action_dim, max_action)
actor.load_state_dict(state['actor'])
actor.eval()

# Run actor on dataset observations and compare to dataset actions
obs = qdataset['observations']
# If observations were normalized in training, the config may have printed mean/std — assume not
with torch.no_grad():
    obs_t = torch.tensor(obs, dtype=torch.float32)
    pred = actor(obs_t).cpu().numpy().flatten()

# If actions were 1D, qdataset['actions'] shape may be (N,) or (N,1)
acts = qdataset['actions']
acts = acts.flatten()

mse = np.mean((pred - acts) ** 2)
print(f'Pred actions MSE on dataset: {mse:.6f}')

# Save predictions to a CSV
out_csv = os.path.join(CHECKPOINT_DIR, 'predicted_actions.csv')
np.savetxt(out_csv, pred, delimiter=',')
print('Saved predicted actions to', out_csv)
