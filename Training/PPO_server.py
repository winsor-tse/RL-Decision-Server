# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
    """whether to save the PPO agent checkpoint"""
    model_path: str | None = None
    """checkpoint override; defaults to runs/<run_name>/PPO_server.pt"""
    restore_model_path: str | None = None
    """PyTorch checkpoint whose agent weights initialize this training run"""

    # Algorithm specific arguments
    total_timesteps: int = 20000
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


def save_agent(agent: nn.Module, model_path: str | Path) -> Path:
    """Save the PPO actor and critic state to a deterministic checkpoint path."""
    checkpoint_path = Path(model_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), checkpoint_path)
    return checkpoint_path


def restore_agent(
    agent: nn.Module,
    restore_model_path: str | Path,
    device: torch.device,
) -> Path:
    """Restore PPO agent weights from an existing PyTorch checkpoint."""
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


def create_run_paths(
    args: Args,
    *,
    timestamp: int | None = None,
    runs_directory: str | Path = "runs",
) -> tuple[str, Path, Path]:
    """Create one run directory and resolve its PyTorch checkpoint path."""
    run_timestamp = int(time.time()) if timestamp is None else timestamp
    run_name = f"Env16__{args.exp_name}__{args.seed}__{run_timestamp}"
    run_directory = Path(runs_directory) / run_name
    checkpoint_path = training_checkpoint_path(
        run_directory,
        args.model_path,
        "PPO_server.pt",
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    args.model_path = str(checkpoint_path)
    return run_name, run_directory, checkpoint_path


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name, run_directory, checkpoint_path = create_run_paths(args)
    print(f"Run directory: {run_directory.resolve()}", flush=True)
    print(f"Model checkpoint: {checkpoint_path.resolve()}", flush=True)

    writer = SummaryWriter(str(run_directory))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = Env16()

    agent = Agent(envs).to(device)
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

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.as_tensor(
        next_obs,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    next_done = torch.zeros(args.num_envs).to(device)
    episode_return = 0.0
    episode_length = 0
    episode_reward_components = {}
    completed_episodes = 0
    wins = 0
    action_counts = np.zeros(envs.single_action_space.n, dtype=np.int64)
    recent_actions = []
    player_hp_index = 3
    enemy_hp_index = int(envs.config["OBS_PLAYER_SIZE"]) + 2

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            action_index = int(action.item())
            action_counts[action_index] += 1
            recent_actions.append(action_index)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_observation, reward, termination, truncation, info = envs.step(
                action.cpu().numpy()
            )
            done = bool(termination or truncation)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
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
                    action_names=envs.Actions,
                    recent_actions=recent_actions,
                )
                recent_actions.clear()

            if done:
                outcome = info.get("episode_outcome")
                is_win = outcome == "win"
                completed_episodes += 1
                wins += int(is_win)
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
                    f"length={episode_length}, return={episode_return:.2f}, "
                    f"win_rate={win_percentage:.2f}%"
                )
                episode_return = 0.0
                episode_length = 0
                episode_reward_components.clear()
                next_observation, _ = envs.reset()

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

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                #TD Error
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                #GAE advantage
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            #Return (critic target)
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                #Probability ratio
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    #KL divergence, Gradient is not backpropped
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # Clipped surrogate objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # MSE
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                # Total PPO Loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.save_model:
            saved_checkpoint = save_agent(agent, checkpoint_path)
            print(f"model saved to {saved_checkpoint}")

    envs.close()
    writer.close()
