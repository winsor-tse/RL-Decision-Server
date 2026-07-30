# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Utils.buffers import ReplayBuffer
from Custom_enviornments.Test_Env import Env_16


DEFAULT_TOTAL_TIMESTEPS = 20_000
CLEANRL_REFERENCE_TOTAL_TIMESTEPS = 500_000
CLEANRL_COUNT_RATIOS = {
    "buffer_size": 10_000 / CLEANRL_REFERENCE_TOTAL_TIMESTEPS,
    "target_network_frequency": 500 / CLEANRL_REFERENCE_TOTAL_TIMESTEPS,
    "batch_size": 128 / CLEANRL_REFERENCE_TOTAL_TIMESTEPS,
    "learning_starts": 10_000 / CLEANRL_REFERENCE_TOTAL_TIMESTEPS,
    "train_frequency": 10 / CLEANRL_REFERENCE_TOTAL_TIMESTEPS,
    "metrics_frequency": 10 / CLEANRL_REFERENCE_TOTAL_TIMESTEPS,
}


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = True
    """save model is defaulted as True"""

    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    """total number of environment steps"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 400
    """replay-buffer capacity"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 20
    """target-network update interval"""
    batch_size: int = 5
    """training batch size"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 400
    """environment step after which learning begins"""
    train_frequency: int = 1
    """environment steps between training updates"""
    metrics_frequency: int = 1
    """environment steps between regular TensorBoard writes"""

    def __post_init__(self) -> None:
        """Warn about unusual values without modifying user configuration."""
        if self.total_timesteps <= 0:
            warnings.warn(
                "total_timesteps should be greater than zero",
                UserWarning,
                stacklevel=2,
            )
            return

        for name, ratio in CLEANRL_COUNT_RATIOS.items():
            value = getattr(self, name)
            recommended = max(1, int(round(self.total_timesteps * ratio)))
            lower_bound = max(1, int(round(recommended * 0.5)))
            upper_bound = max(lower_bound, int(round(recommended * 2.0)))
            if value < lower_bound or value > upper_bound:
                warnings.warn(
                    f"{name}={value} is outside the recommended range "
                    f"{lower_bound}-{upper_bound} for "
                    f"total_timesteps={self.total_timesteps} "
                    f"(CleanRL-ratio target: {recommended})",
                    UserWarning,
                    stacklevel=2,
                )

        if self.batch_size > self.buffer_size:
            warnings.warn(
                "batch_size exceeds buffer_size; replay samples will contain "
                "heavy duplication",
                UserWarning,
                stacklevel=2,
            )
        if self.learning_starts >= self.total_timesteps:
            warnings.warn(
                "learning_starts is at or beyond total_timesteps; no training "
                "updates will occur",
                UserWarning,
                stacklevel=2,
            )


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def log_step_metrics(
    writer,
    *,
    global_step: int,
    reward: float,
    epsilon: float,
    replay_buffer_size: int,
    observation: np.ndarray,
    player_hp_index: int,
    enemy_hp_index: int,
    reward_components: dict[str, float],
    action_counts: np.ndarray,
    action_names: list[str],
    recent_actions: list[int],
) -> None:
    """Write sampled environment and training-state metrics to TensorBoard."""
    writer.add_scalar("charts/step_reward", reward, global_step)
    writer.add_scalar("charts/epsilon", epsilon, global_step)
    writer.add_scalar(
        "training/replay_buffer_size",
        replay_buffer_size,
        global_step,
    )
    writer.add_scalar(
        "environment/player_hp",
        float(observation[player_hp_index]),
        global_step,
    )
    writer.add_scalar(
        "environment/enemy_hp",
        float(observation[enemy_hp_index]),
        global_step,
    )
    for component_name, component_value in reward_components.items():
        writer.add_scalar(
            f"rewards/{component_name}",
            float(component_value),
            global_step,
        )

    total_actions = int(action_counts.sum())
    if total_actions:
        for index, action_name in enumerate(action_names):
            metric_name = action_name.replace(":", "_").replace("/", "_")
            writer.add_scalar(
                f"environment/action_frequency/{metric_name}",
                action_counts[index] / total_actions,
                global_step,
            )
    if recent_actions:
        writer.add_histogram(
            "environment/action_frequency",
            np.asarray(recent_actions, dtype=np.int64),
            global_step,
            bins=np.arange(len(action_names) + 1) - 0.5,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    from torch.utils.tensorboard import SummaryWriter

    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.exp_name}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    hyperparameters = {
        **vars(args),
        "cleanrl_reference_total_timesteps": CLEANRL_REFERENCE_TOTAL_TIMESTEPS,
    }
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(
            f"|{key}|{value}|" for key, value in hyperparameters.items()
        ),
    )
    configured_names = (
        "buffer_size",
        "target_network_frequency",
        "batch_size",
        "learning_starts",
        "train_frequency",
        "metrics_frequency",
    )
    print(
        "Configured hyperparameters: "
        + ", ".join(f"{name}={getattr(args, name)}" for name in configured_names)
    )

    #TODO: determine whether the following is needed
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = Env_16.Env16()

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    episode_return = 0.0
    episode_length = 0
    episode_reward_components = {}
    completed_episodes = 0
    wins = 0
    action_counts = np.zeros(envs.single_action_space.n, dtype=np.int64)
    recent_actions = []
    metrics_frequency = args.metrics_frequency
    player_hp_index = 3
    enemy_hp_index = int(envs.config["OBS_PLAYER_SIZE"]) + 2
    #obs = envs.next_state #intialize as zero first
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                q_values = q_network(torch.as_tensor(obs, dtype=torch.float32, device=device))
                actions = np.array([torch.argmax(q_values).item()])
        action_index = int(np.asarray(actions).item())
        action_counts[action_index] += 1
        recent_actions.append(action_index)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        episode_return += float(rewards)
        episode_length += 1
        reward_components = infos.get("reward_components", {})
        for component_name, component_value in reward_components.items():
            episode_reward_components[component_name] = (
                episode_reward_components.get(component_name, 0.0)
                + float(component_value)
            )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if truncations:
            real_next_obs = infos.get("next_state", real_next_obs)
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # Gymnasium.vector.SyncVectorEnv or AsyncVectorEnv which handle environment auto-resetting automatically. we need to Manually reset here for external envs.
        done = bool(terminations or truncations)
        has_reward_event = any(
            float(component_value) != 0.0
            for component_value in reward_components.values()
        )
        if global_step % metrics_frequency == 0 or done or has_reward_event:
            log_step_metrics(
                writer,
                global_step=global_step,
                reward=float(rewards),
                epsilon=epsilon,
                replay_buffer_size=rb.size(),
                observation=next_obs,
                player_hp_index=player_hp_index,
                enemy_hp_index=enemy_hp_index,
                reward_components=reward_components,
                action_counts=action_counts,
                action_names=envs.Actions,
                recent_actions=recent_actions,
            )
            recent_actions.clear()

        if done:
            outcome = infos.get("episode_outcome")
            is_win = outcome == "win"
            completed_episodes += 1
            wins += int(is_win)
            win_percentage = 100.0 * wins / completed_episodes

            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episode_length", episode_length, global_step)
            writer.add_scalar(
                "charts/win_rate",
                win_percentage,
                global_step,
            )
            for component_name, component_total in episode_reward_components.items():
                writer.add_scalar(
                    f"rewards/episode_{component_name}",
                    component_total,
                    global_step,
                )

            print(
                f"episode={completed_episodes}, outcome={outcome}, "
                f"length={episode_length}, return={episode_return:.2f}, "
                f"win_rate={win_percentage:.2f}%"
            )
            episode_return = 0.0
            episode_length = 0
            episode_reward_components.clear()
            obs, _ = envs.reset()
        else:
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    #DQN's Bellmans
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # Log loss metrics every train step
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/mean_q_value", old_val.mean().item(), global_step)

                if global_step % 100 == 0 and args.save_model:                    
                    #Save Model
                    model_path = f"runs/{run_name}/{args.exp_name}.pt"
                    torch.save(q_network.state_dict(), model_path)
                    torch.onnx.export #export to onnx
                    print(f"model saved to {model_path}")

                # optimize the model
                optimizer.zero_grad() # reset grad
                loss.backward() #back prop
                optimizer.step() # learning rate * gradients

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    envs.close()
    writer.close()
