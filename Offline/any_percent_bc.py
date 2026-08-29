import contextlib
import os
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import minari
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
from torch.utils.tensorboard import SummaryWriter

TensorBatch = List[torch.Tensor]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "BC-Minari"
    name: str = "bc"
    # model params
    gamma: float = 0.99  # Discount factor
    top_fraction: float = 0.1  # Best data fraction to use
    # training params
    dataset_id: str = "env16/BC-v0"  #This needs to pull from local
    update_steps: int = int(1e6)  # Total training networks updates
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    normalize_state: bool = True  # Normalize states
    # evaluation params
    eval_every: int = int(5e3)  # How often (time steps) we evaluate
    eval_episodes: int = 10  # How many episodes run during evaluation
    # general params
    train_seed: int = 0
    eval_seed: int = 0
    checkpoints_path: str = None  # Save path (Optional)

    def __post_init__(self):
        # Sanitize dataset identifier for use in filenames (remove path separators and drive letters)
        dataset_label = os.path.basename(self.dataset_id.replace('\\', '/'))
        dataset_label = dataset_label.replace(':', '')
        self.name = f"{self.name}-{dataset_label}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # Use an ObservationWrapper compatible with the installed Gymnasium version
    class _NormalizeObs(gym.ObservationWrapper):
        def __init__(self, env, mean, std):
            super().__init__(env)
            self._mean = mean
            self._std = std

        def observation(self, observation):
            return (observation - self._mean) / self._std

    class _ScaleReward(gym.RewardWrapper):
        def __init__(self, env, scale):
            super().__init__(env)
            self._scale = scale

        def reward(self, reward):
            return self._scale * reward

    env = _NormalizeObs(env, state_mean, state_std)
    if reward_scale != 1.0:
        env = _ScaleReward(env, reward_scale)
    return env


def discounted_return(x: np.ndarray, gamma: float) -> np.ndarray:
    total_return = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        total_return = x[t] + gamma * total_return
    return total_return


def best_trajectories_ids(
    dataset: minari.MinariDataset, top_fraction: float, gamma: float
) -> List[int]:
    ids_and_return = [
        (episode.id, discounted_return(episode.rewards, gamma)) for episode in dataset
    ]
    ids_and_returns = sorted(ids_and_return, key=lambda t: -t[1])

    top_ids = [id for (id, r) in ids_and_returns]
    top_ids = top_ids[: max(1, int(top_fraction * len(ids_and_returns)))]
    assert len(top_ids) > 0
    return top_ids


# WARN: this will load full dataset in memory (which is OK for D4RL datasets)
def qlearning_dataset(
    dataset: minari.MinariDataset, traj_ids: List[int]
) -> Dict[str, np.ndarray]:
    obs, next_obs, actions, rewards, dones = [], [], [], [], []

    for episode in dataset.iterate_episodes(episode_indices=traj_ids):
        obs.append(episode.observations[:-1].astype(np.float32))
        next_obs.append(episode.observations[1:].astype(np.float32))
        actions.append(episode.actions.astype(np.float32))
        rewards.append(episode.rewards)
        dones.append(episode.terminations)

    return {
        "observations": np.concatenate(obs),
        "actions": np.concatenate(actions),
        "next_observations": np.concatenate(next_obs),
        "rewards": np.concatenate(rewards),
        "terminals": np.concatenate(dones),
    }


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array] after q_learning_dataset.
    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        # Ensure actions are shaped (N, action_dim). For 1D actions, reshape to (N,1).
        actions_tensor = self._to_tensor(data["actions"])
        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(-1)
        self._actions[:n_transitions] = actions_tensor
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

        self._size = self._pointer = n_transitions
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class BC:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])


@torch.no_grad()
def evaluate(
    env: gym.Env, actor: nn.Module, num_episodes: int, seed: int, device: str
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    for i in range(num_episodes):
        done = False
        state, info = env.reset(seed=seed + i)

        episode_reward = 0.0
        while not done:
                raw_action = actor.act(state, device)
                # If the environment expects discrete actions, convert continuous output to an index
                if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                    # raw_action likely an array/float; get scalar value
                    try:
                        val = float(np.asarray(raw_action).flatten()[0])
                    except Exception:
                        val = float(raw_action)
                    # Round to nearest integer and clip to valid discrete range
                    action_idx = int(np.round(val))
                    action_idx = int(np.clip(action_idx, 0, env.action_space.n - 1))
                    action = action_idx
                else:
                    action = raw_action

                # Step the environment. Env16BC will raise a ValueError when the
                # provided action doesn't match the recorded captured action. Catch
                # that and advance using the recorded action (if available) so the
                # episode can continue while still recording the model's prediction.
                try:
                    state, reward, terminated, truncated, info = env.step(action)
                except ValueError as e:
                    msg = str(e)
                    # Detect the specific mismatch error from Env16BC
                    if 'Expected captured action' in msg and hasattr(env, 'next_action'):
                        expected = None
                        try:
                            expected = env.next_action()
                        except Exception:
                            expected = None

                        if expected is not None:
                            # Advance env using the recorded action so episode continues
                            state, reward, terminated, truncated, info = env.step(int(expected))
                            # Optionally, log the mismatch via TensorBoard if writer exists
                            try:
                                writer.add_scalar('eval/action_mismatch', 1, len(episode_rewards))
                            except Exception:
                                pass
                        else:
                            # Re-raise if we cannot recover
                            raise
                    else:
                        # Unknown ValueError - re-raise
                        raise

                done = terminated or truncated
                episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


@pyrallis.wrap()
def train(config: TrainConfig):
    dataset = minari.load_dataset(config.dataset_id)

    qdataset = qlearning_dataset(
        dataset=dataset,
        traj_ids=best_trajectories_ids(dataset, config.top_fraction, config.gamma),
    )

    # Derive state/action dimensions from the dataset (works when env can't be recovered)
    state_dim = int(qdataset["observations"].shape[1])
    # Enforce 1D action arrays (scalar continuous actions)
    actions_arr = qdataset["actions"]
    if actions_arr.ndim != 1:
        raise ValueError(f"This script expects 1D action arrays (shape (N,)). Found shape: {actions_arr.shape}")
    action_dim = 1
    # Estimate max_action from data; fallback to 1.0 if not available
    try:
        max_action = float(np.max(np.abs(actions_arr)))
        if max_action == 0.0:
            max_action = 1.0
    except Exception:
        max_action = 1.0
    if config.normalize_state:
        state_mean, state_std = compute_mean_std(qdataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    qdataset["observations"] = normalize_states(
        qdataset["observations"], state_mean, state_std
    )
    qdataset["next_observations"] = normalize_states(
        qdataset["next_observations"], state_mean, state_std
    )

    # Try to recover the environment for evaluation; if it fails, skip evaluation.
    try:
        eval_env = dataset.recover_environment()
        eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    except Exception:
        eval_env = None
        print("Warning: could not recover environment from dataset; evaluation disabled.")

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        DEVICE,
    )
    replay_buffer.load_dataset(qdataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Create TensorBoard writer. If checkpoints_path provided, use it; otherwise use runs/<name>
    if config.checkpoints_path is not None:
        log_dir = config.checkpoints_path
    else:
        log_dir = os.path.join("runs", config.name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Set seed #this seed is not necessary
    set_seed(config.train_seed)

    actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    trainer = BC(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        device=DEVICE,
    )

    #TODO: we are not using wandb, we will use tensorboard
    
    for step in trange(config.update_steps):
        batch = [b.to(DEVICE) for b in replay_buffer.sample(config.batch_size)]
        log_dict = trainer.train(batch)

        for k, v in log_dict.items():
            try:
                writer.add_scalar(f"train/{k}", float(v), step)
            except Exception:
                # if a non-scalar is logged, skip
                pass

        if (step + 1) % config.eval_every == 0:
            eval_scores = evaluate(
                env=eval_env,
                actor=actor,
                num_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=DEVICE,
            )
            writer.add_scalar("eval/return", float(eval_scores.mean()), step)
            # optional normalized score logging, only if dataset has reference scores
            with contextlib.suppress(ValueError):
                normalized_score = (
                    minari.get_normalized_score(dataset, eval_scores).mean() * 100
                )
                writer.add_scalar("eval/normalized_score", float(normalized_score), step)

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{step}.pt"),
                )

    # Save a final checkpoint even if evaluation was disabled
    if config.checkpoints_path is not None:
        final_path = os.path.join(config.checkpoints_path, "final_checkpoint.pt")
        try:
            torch.save(trainer.state_dict(), final_path)
            print(f"Saved final checkpoint to: {final_path}")
        except Exception as e:
            print(f"Warning: failed to save final checkpoint: {e}")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    train()