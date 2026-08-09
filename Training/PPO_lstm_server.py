# Based on CleanRL's recurrent PPO implementation:
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from Custom_enviornments.Test_Env.Env_16 import Env16
from Training.ppo_metrics import log_step_metrics
from Utils.model_paths import training_checkpoint_path


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
    """whether to save the recurrent PPO agent checkpoint"""
    model_path: str | None = None
    """checkpoint override; defaults to runs/<run_name>/PPO_lstm_server.pt"""
    restore_model_path: str | None = None
    """PyTorch checkpoint whose agent weights initialize this training run"""

    # Algorithm specific arguments
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of live game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    metrics_frequency: int = 1
    """environment steps between regular TensorBoard writes"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def recurrent_minibatches(
    num_steps: int,
    num_minibatches: int,
) -> list[np.ndarray]:
    """Return shuffled, contiguous sequences for a single recurrent env."""
    if num_minibatches <= 0:
        raise ValueError("num_minibatches must be greater than zero")
    if num_steps % num_minibatches != 0:
        raise ValueError("num_steps must be divisible by num_minibatches")

    sequence_length = num_steps // num_minibatches
    starts = np.arange(0, num_steps, sequence_length)
    np.random.shuffle(starts)
    return [
        np.arange(start, start + sequence_length, dtype=np.int64)
        for start in starts
    ]


def save_agent(agent: nn.Module, model_path: str) -> Path:
    """Save the recurrent actor and critic to a deterministic checkpoint."""
    checkpoint_path = Path(model_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), checkpoint_path)
    return checkpoint_path


def restore_agent(
    agent: nn.Module,
    restore_model_path: str | Path,
    device: torch.device,
) -> Path:
    """Restore recurrent PPO weights from an existing PyTorch checkpoint."""
    checkpoint_path = Path(restore_model_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Restore checkpoint does not exist: {checkpoint_path}"
        )
    agent.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    return checkpoint_path


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.obs_dim = int(np.prod(envs.single_observation_space.shape))
        hidden_size = self.obs_dim * envs.single_action_space.n
        self.network = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 512)),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, parameter in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(parameter, 0)
            elif "weight" in name:
                nn.init.orthogonal_(parameter, 1.0)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_states(self, x, lstm_state, done):
        embedding = self.network(x)
        batch_size = lstm_state[0].shape[1]
        embedding = embedding.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        embeddings = []
        for step_embedding, step_done in zip(embedding, done):
            step_embedding, lstm_state = self.lstm(
                step_embedding.unsqueeze(0),
                (
                    (1.0 - step_done).view(1, -1, 1) * lstm_state[0],
                    (1.0 - step_done).view(1, -1, 1) * lstm_state[1],
                ),
            )
            embeddings.append(step_embedding)
        return torch.flatten(torch.cat(embeddings), 0, 1), lstm_state

    def get_value(self, x, lstm_state, done):
        embedding, _ = self.get_states(x, lstm_state, done)
        return self.critic(embedding)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        embedding, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(embedding)
        probabilities = Categorical(logits=logits)
        if action is None:
            action = probabilities.sample()
        return (
            action,
            probabilities.log_prob(action),
            probabilities.entropy(),
            self.critic(embedding),
            lstm_state,
        )


def train(args: Args) -> None:
    if args.num_envs != 1:
        raise ValueError("Env16 supports exactly one live environment")
    if args.metrics_frequency <= 0:
        raise ValueError("metrics_frequency must be greater than zero")
    if args.num_steps <= 0:
        raise ValueError("num_steps must be greater than zero")
    if args.num_minibatches <= 0:
        raise ValueError("num_minibatches must be greater than zero")
    if args.update_epochs <= 0:
        raise ValueError("update_epochs must be greater than zero")
    if args.num_steps % args.num_minibatches != 0:
        raise ValueError("num_steps must be divisible by num_minibatches")

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"Env16__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_directory = Path("runs") / run_name
    args.model_path = str(
        training_checkpoint_path(
            run_directory,
            args.model_path,
            "PPO_lstm_server.pt",
        )
    )

    writer = SummaryWriter(str(run_directory))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    print(f"Using device: {device}", flush=True)
    print(
        "Initializing Env16; waiting for the first game ai_tick...",
        flush=True,
    )

    env = None
    try:
        env = Env16()
        agent = Agent(env).to(device)
        if args.restore_model_path:
            restored_checkpoint = restore_agent(
                agent,
                args.restore_model_path,
                device,
            )
            print(
                f"Restored model from {restored_checkpoint.resolve()}",
                flush=True,
            )
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        obs = torch.zeros(
            (args.num_steps, args.num_envs)
            + env.single_observation_space.shape,
            device=device,
        )
        actions = torch.zeros(
            (args.num_steps, args.num_envs),
            dtype=torch.long,
            device=device,
        )
        logprobs = torch.zeros(
            (args.num_steps, args.num_envs),
            device=device,
        )
        rewards = torch.zeros(
            (args.num_steps, args.num_envs),
            device=device,
        )
        dones = torch.zeros(
            (args.num_steps, args.num_envs),
            device=device,
        )
        values = torch.zeros(
            (args.num_steps, args.num_envs),
            device=device,
        )
        lstm_hidden = torch.zeros(
            (
                args.num_steps,
                agent.lstm.num_layers,
                args.num_envs,
                agent.lstm.hidden_size,
            ),
            device=device,
        )
        lstm_cell = torch.zeros_like(lstm_hidden)

        global_step = 0
        start_time = time.time()
        next_observation, _ = env.reset(seed=args.seed)
        next_obs = torch.as_tensor(
            next_observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        next_done = torch.zeros(args.num_envs, device=device)
        next_lstm_state = (
            torch.zeros(
                agent.lstm.num_layers,
                args.num_envs,
                agent.lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                agent.lstm.num_layers,
                args.num_envs,
                agent.lstm.hidden_size,
                device=device,
            ),
        )

        episode_return = 0.0
        episode_length = 0
        episode_reward_components: dict[str, float] = {}
        completed_episodes = 0
        wins = 0
        action_counts = np.zeros(env.single_action_space.n, dtype=np.int64)
        recent_actions: list[int] = []
        player_hp_index = 3
        enemy_hp_index = int(env.config["OBS_PLAYER_SIZE"]) + 2

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                fraction = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = (
                    fraction * args.learning_rate
                )

            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                lstm_hidden[step] = next_lstm_state[0]
                lstm_cell[step] = next_lstm_state[1]

                with torch.no_grad():
                    (
                        action,
                        logprob,
                        _,
                        value,
                        next_lstm_state,
                    ) = agent.get_action_and_value(
                        next_obs,
                        next_lstm_state,
                        next_done,
                    )
                    values[step] = value.flatten()
                    writer.add_scalar(
                        "lstm/hidden_state_norm",
                        next_lstm_state[0].norm().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "lstm/cell_state_norm",
                        next_lstm_state[1].norm().item(),
                        global_step,
                    )
                actions[step] = action
                logprobs[step] = logprob

                action_index = int(action.item())
                action_counts[action_index] += 1
                recent_actions.append(action_index)
                (
                    next_observation,
                    reward,
                    termination,
                    truncation,
                    info,
                ) = env.step(action.cpu().numpy())
                done = bool(termination or truncation)
                rewards[step] = torch.as_tensor(
                    [reward],
                    dtype=torch.float32,
                    device=device,
                )
                episode_return += float(reward)
                episode_length += 1

                reward_components = info.get("reward_components", {})
                for component_name, component_value in reward_components.items():
                    episode_reward_components[component_name] = (
                        episode_reward_components.get(component_name, 0.0)
                        + float(component_value)
                    )

                has_reward_event = any(
                    float(component_value) != 0.0
                    for component_value in reward_components.values()
                )
                if (
                    global_step % args.metrics_frequency == 0
                    or done
                    or has_reward_event
                ):
                    log_step_metrics(
                        writer,
                        global_step=global_step,
                        reward=float(reward),
                        observation=next_observation,
                        player_hp_index=player_hp_index,
                        enemy_hp_index=enemy_hp_index,
                        reward_components=reward_components,
                        action_counts=action_counts,
                        action_names=env.Actions,
                        recent_actions=recent_actions,
                    )
                    recent_actions.clear()

                if done:
                    outcome = info.get("episode_outcome")
                    completed_episodes += 1
                    wins += int(outcome == "win")
                    win_percentage = 100.0 * wins / completed_episodes
                    writer.add_scalar(
                        "charts/episodic_return",
                        episode_return,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episode_length",
                        episode_length,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/win_rate",
                        win_percentage,
                        global_step,
                    )
                    for component_name, component_total in (
                        episode_reward_components.items()
                    ):
                        writer.add_scalar(
                            f"rewards/episode_{component_name}",
                            component_total,
                            global_step,
                        )

                    print(
                        f"episode={completed_episodes}, outcome={outcome}, "
                        f"length={episode_length}, "
                        f"return={episode_return:.2f}, "
                        f"win_rate={win_percentage:.2f}%"
                    )
                    episode_return = 0.0
                    episode_length = 0
                    episode_reward_components.clear()
                    next_observation, _ = env.reset()

                next_obs = torch.as_tensor(
                    next_observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                next_done = torch.as_tensor(
                    [done],
                    dtype=torch.float32,
                    device=device,
                )

            with torch.no_grad():
                next_value = agent.get_value(
                    next_obs,
                    next_lstm_state,
                    next_done,
                ).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                last_gae_lambda = 0
                for step in reversed(range(args.num_steps)):
                    if step == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones[step + 1]
                        next_values = values[step + 1]
                    delta = (
                        rewards[step]
                        + args.gamma * next_values * next_nonterminal
                        - values[step]
                    )
                    last_gae_lambda = (
                        delta
                        + args.gamma
                        * args.gae_lambda
                        * next_nonterminal
                        * last_gae_lambda
                    )
                    advantages[step] = last_gae_lambda
                returns = advantages + values

            b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_dones = dones.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            clip_fractions = []
            for _ in range(args.update_epochs):
                for minibatch_indices in recurrent_minibatches(
                    args.num_steps,
                    args.num_minibatches,
                ):
                    sequence_start = int(minibatch_indices[0])
                    initial_lstm_state = (
                        lstm_hidden[sequence_start],
                        lstm_cell[sequence_start],
                    )
                    (
                        _,
                        new_logprob,
                        entropy,
                        new_value,
                        _,
                    ) = agent.get_action_and_value(
                        b_obs[minibatch_indices],
                        initial_lstm_state,
                        b_dones[minibatch_indices],
                        b_actions[minibatch_indices],
                    )
                    log_ratio = (
                        new_logprob - b_logprobs[minibatch_indices]
                    )
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fractions.append(
                            (
                                (ratio - 1.0).abs() > args.clip_coef
                            ).float().mean().item()
                        )

                    minibatch_advantages = b_advantages[minibatch_indices]
                    if args.norm_adv:
                        minibatch_advantages = (
                            minibatch_advantages
                            - minibatch_advantages.mean()
                        ) / (minibatch_advantages.std() + 1e-8)

                    policy_loss_unclipped = -minibatch_advantages * ratio
                    policy_loss_clipped = -minibatch_advantages * torch.clamp(
                        ratio,
                        1 - args.clip_coef,
                        1 + args.clip_coef,
                    )
                    policy_loss = torch.max(
                        policy_loss_unclipped,
                        policy_loss_clipped,
                    ).mean()

                    new_value = new_value.view(-1)
                    if args.clip_vloss:
                        value_loss_unclipped = (
                            new_value - b_returns[minibatch_indices]
                        ) ** 2
                        value_clipped = b_values[
                            minibatch_indices
                        ] + torch.clamp(
                            new_value - b_values[minibatch_indices],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        value_loss_clipped = (
                            value_clipped - b_returns[minibatch_indices]
                        ) ** 2
                        value_loss = 0.5 * torch.max(
                            value_loss_unclipped,
                            value_loss_clipped,
                        ).mean()
                    else:
                        value_loss = 0.5 * (
                            (new_value - b_returns[minibatch_indices]) ** 2
                        ).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        policy_loss
                        - args.ent_coef * entropy_loss
                        + args.vf_coef * value_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.parameters(),
                        args.max_grad_norm,
                    )
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            predicted_values = b_values.cpu().numpy()
            target_values = b_returns.cpu().numpy()
            target_variance = np.var(target_values)
            explained_variance = (
                np.nan
                if target_variance == 0
                else 1
                - np.var(target_values - predicted_values) / target_variance
            )
            writer.add_scalar(
                "charts/learning_rate",
                optimizer.param_groups[0]["lr"],
                global_step,
            )
            writer.add_scalar(
                "losses/value_loss",
                value_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "losses/policy_loss",
                policy_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "losses/entropy",
                entropy_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "losses/old_approx_kl",
                old_approx_kl.item(),
                global_step,
            )
            writer.add_scalar(
                "losses/approx_kl",
                approx_kl.item(),
                global_step,
            )
            writer.add_scalar(
                "losses/clipfrac",
                np.mean(clip_fractions),
                global_step,
            )
            writer.add_scalar(
                "losses/explained_variance",
                explained_variance,
                global_step,
            )
            steps_per_second = int(global_step / (time.time() - start_time))
            print("SPS:", steps_per_second)
            writer.add_scalar("charts/SPS", steps_per_second, global_step)

            if args.save_model:
                checkpoint_path = save_agent(agent, args.model_path)
                print(f"model saved to {checkpoint_path}")
    finally:
        if env is not None:
            env.close()
        writer.close()


def main() -> None:
    train(tyro.cli(Args))


if __name__ == "__main__":
    main()
