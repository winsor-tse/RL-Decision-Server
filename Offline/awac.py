"""Discrete AWAC for Env16, using local BC-Minari demonstrations."""

import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import minari
import numpy as np
import torch
import torch.nn as nn
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from Custom_enviornments.Load_env_config import load_env_config
from Custom_enviornments.Test_Env.Env_16 import Env16
from Custom_enviornments.Test_Env.Env_16_BC import BC_ACTIONS_11

TensorBatch = List[torch.Tensor]
AWAC_MODEL_FILENAME = "AWAC_model.pt"


@dataclass
class TrainConfig:
    name: str = "AWAC"
    dataset_id: str = "env16/BC-v0"  # Local BC-Minari dataset, as in any_percent_bc.
    checkpoints_path: str = "runs"
    seed: int = 42
    deterministic_torch: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    buffer_size: int = 2_000_000
    offline_iterations: int = 1_000_000
    online_iterations: int = 0  # Opt in; requires the bridge and running game.
    batch_size: int = 256
    normalize_state: bool = True
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0

    def __post_init__(self):
        if self.offline_iterations < 0 or self.online_iterations < 0:
            raise ValueError("Training iteration counts must be nonnegative")
        if min(self.buffer_size, self.batch_size, self.hidden_dim) <= 0:
            raise ValueError("Buffer, batch, and hidden sizes must be positive")
        if self.awac_lambda <= 0 or self.learning_rate <= 0:
            raise ValueError("awac_lambda and learning_rate must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.tau <= 1:
            raise ValueError("gamma and tau must be between 0 and 1")
        label = os.path.basename(self.dataset_id.replace("\\", "/")).replace(":", "")
        self.name = f"{self.name}-{label}-{str(uuid.uuid4())[:8]}"
        self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def qlearning_dataset(dataset) -> Dict[str, np.ndarray]:
    """Keep all demonstrations and bootstrap across time-limit truncations."""
    if not isinstance(dataset.action_space, gym.spaces.Discrete):
        raise ValueError("Env16 requires a discrete Minari action space")
    if dataset.action_space.n != len(BC_ACTIONS_11) or dataset.action_space.start != 0:
        raise ValueError("Dataset must use the 11 zero-based BC action indices")
    fields = {key: [] for key in
              ("observations", "actions", "next_observations", "rewards", "terminals")}
    obs_size = load_env_config()["OBS_SIZE"]
    for episode in dataset.iterate_episodes():
        actions = np.asarray(episode.actions)
        if not len(actions):
            continue
        obs = np.asarray(episode.observations, dtype=np.float32)
        if obs.shape != (len(actions) + 1, obs_size):
            raise ValueError(f"Expected Env16 observations shaped (N+1, {obs_size}); got {obs.shape}")
        if actions.shape != (len(actions),) or not np.all(
            np.isfinite(actions) & (actions == np.floor(actions))
            & (actions >= 0) & (actions < len(BC_ACTIONS_11))
        ):
            raise ValueError("Dataset contains invalid discrete action indices")
        fields["observations"].append(obs[:-1])
        fields["next_observations"].append(obs[1:])
        fields["actions"].append(actions)
        fields["rewards"].append(np.asarray(episode.rewards, dtype=np.float32))
        fields["terminals"].append(np.asarray(episode.terminations, dtype=np.float32))
    if not fields["actions"]:
        raise ValueError("The local Minari dataset contains no transitions")
    return {key: np.concatenate(values) for key, values in fields.items()}


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

    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"]).reshape(-1, 1)
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = self._size % self._buffer_size

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


class Actor(nn.Module):
    """Categorical policy over the BC action ordering."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _get_policy(self, state):
        return torch.distributions.Categorical(logits=self._mlp(state))

    def log_prob(self, state, action):
        return self._get_policy(state).log_prob(action.long().reshape(-1)).unsqueeze(-1)

    def forward(self, state):
        policy = self._get_policy(state)
        action = policy.sample()
        return action.unsqueeze(-1), policy.log_prob(action).unsqueeze(-1)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> int:
        state = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
        policy = self._get_policy(state)
        action = policy.sample() if self.training else policy.probs.argmax(-1)
        return int(action.item())


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, action=None):
        values = self._mlp(state)
        return values if action is None else values.gather(1, action.long().reshape(-1, 1))


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            policy = self._actor._get_policy(states)
            v = (policy.probs * torch.min(
                self._critic_1(states), self._critic_2(states)
            )).sum(-1, keepdim=True)

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            policy = self._actor._get_policy(next_states)
            q_next = (policy.probs * torch.min(
                self._target_critic_1(next_states),
                self._target_critic_2(next_states),
            )).sum(-1, keepdim=True)
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
    ):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
            "target_critic_1": self._target_critic_1.state_dict(),
            "target_critic_2": self._target_critic_2.state_dict(),
            "actor_optimizer": self._actor_optimizer.state_dict(),
            "critic_1_optimizer": self._critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self._critic_2_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])
        for name in ("critic_1", "critic_2"):
            getattr(self, "_target_" + name).load_state_dict(
                state_dict.get("target_" + name, state_dict[name])
            )
        for name in ("actor_optimizer", "critic_1_optimizer", "critic_2_optimizer"):
            if name in state_dict:
                getattr(self, "_" + name).load_state_dict(state_dict[name])


def train(config: TrainConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic_torch)
    # Loading metadata/data never recovers Env16BC or opens a live game socket.
    dataset = minari.load_dataset(config.dataset_id, download=False)
    data = qlearning_dataset(dataset)
    state_dim = data["observations"].shape[1]
    if config.normalize_state:
        mean = data["observations"].mean(0)
        std = data["observations"].std(0) + 1e-3
    else:
        mean = np.zeros(state_dim, dtype=np.float32)
        std = np.ones(state_dim, dtype=np.float32)
    for key in ("observations", "next_observations"):
        data[key] = (data[key] - mean) / std
    capacity = config.buffer_size
    if len(data["actions"]) > capacity:
        raise ValueError("Replay buffer is smaller than the dataset")
    if not config.online_iterations:
        capacity = len(data["actions"])
    replay = ReplayBuffer(state_dim, 1, capacity, config.device)
    replay.load_dataset(data)
    del data

    kwargs = dict(state_dim=state_dim, action_dim=len(BC_ACTIONS_11), hidden_dim=config.hidden_dim)
    actor = Actor(**kwargs).to(config.device)
    critic_1 = Critic(**kwargs).to(config.device)
    critic_2 = Critic(**kwargs).to(config.device)
    awac = AdvantageWeightedActorCritic(
        actor, torch.optim.Adam(actor.parameters(), lr=config.learning_rate),
        critic_1, torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate),
        critic_2, torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate),
        gamma=config.gamma, tau=config.tau, awac_lambda=config.awac_lambda,
    )
    os.makedirs(config.checkpoints_path, exist_ok=True)
    with open(os.path.join(config.checkpoints_path, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
    print(f"Checkpoints and TensorBoard: {config.checkpoints_path}")
    writer = SummaryWriter(log_dir=config.checkpoints_path)
    env = None
    completed_steps = 0
    try:
        print("Offline pretraining")
        episode_return, episode_length = 0.0, 0
        for step in trange(config.offline_iterations + config.online_iterations):
            if step == config.offline_iterations:
                print("Online fine-tuning in Env16 (waiting for game ticks)")
                # Match recorded labels: up, left, right, down, ...
                env = Env16(actions=BC_ACTIONS_11)
                raw_state, _ = env.reset(seed=config.seed)
                state = (raw_state - mean) / std
            if step >= config.offline_iterations:
                action = actor.act(state, config.device)
                raw_next, reward, terminated, truncated, _ = env.step(action)
                next_state = (raw_next - mean) / std
                replay.add_transition(state, action, reward, next_state, terminated)
                state = next_state
                episode_return += reward
                episode_length += 1
                if terminated or truncated:
                    writer.add_scalar("online/episode_return", episode_return, step)
                    writer.add_scalar("online/episode_length", episode_length, step)
                    raw_state, _ = env.reset()
                    state = (raw_state - mean) / std
                    episode_return, episode_length = 0.0, 0
            for key, value in awac.update(replay.sample(config.batch_size)).items():
                writer.add_scalar(f"train/{key}", value, step)
            completed_steps = step + 1
    finally:
        try:
            checkpoint = awac.state_dict()
            checkpoint.update(
                state_mean=torch.as_tensor(mean), state_std=torch.as_tensor(std),
                state_dim=state_dim, action_dim=len(BC_ACTIONS_11),
                hidden_dim=config.hidden_dim, actions=list(BC_ACTIONS_11),
                dataset_id=config.dataset_id, completed_steps=completed_steps,
                config=asdict(config),
            )
            model_path = os.path.join(config.checkpoints_path, AWAC_MODEL_FILENAME)
            torch.save(checkpoint, model_path)
            print(f"Saved AWAC model to: {model_path}")
        finally:
            writer.close()
            if env is not None:
                env.close()


if __name__ == "__main__":
    train(tyro.cli(TrainConfig))
