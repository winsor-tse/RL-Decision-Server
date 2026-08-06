import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

from Custom_enviornments.Test_Env.Env_16 import Env16
from Training.PPO_server import Agent


def select_action(
    model: Agent,
    observation: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> int:
    """Select an action from the PPO policy."""
    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        if deterministic:
            logits = model.actor(observation_tensor)
            action = torch.argmax(logits, dim=-1)
        else:
            action, _, _, _ = model.get_action_and_value(observation_tensor)

    return int(action.item())


def evaluate(
    env,
    model: Agent,
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
    deterministic: bool = True,
) -> tuple[list[float], int]:
    """Evaluate a trained PPO policy on the direct external environment."""
    model.eval()
    episodic_returns = []
    wins = 0

    for episode in range(eval_episodes):
        observation, _ = env.reset()
        episode_return = 0.0
        episode_won = False
        done = False
        info = {}

        while not done:
            action = select_action(
                model,
                observation,
                device,
                deterministic,
            )
            observation, reward, terminated, truncated, info = env.step(
                np.array([action])
            )
            episode_return += float(reward)
            done = bool(terminated or truncated)
            if done:
                episode_won = bool(info.get("is_win", False))

        episodic_returns.append(episode_return)
        wins += int(episode_won)
        print(
            f"eval_episode={episode}, episodic_return={episode_return}, "
            f"outcome={'win' if episode_won else info.get('episode_outcome', 'loss')}"
        )

    return episodic_returns, wins


@dataclass
class EvalArgs:
    model_path: str
    """Path to the trained PPO agent checkpoint (.pt file)."""
    eval_episodes: int = 10
    """Number of evaluation episodes to run."""
    deterministic: bool = True
    """Use the highest-probability action instead of sampling the policy."""
    cuda: bool = True
    """Whether to use CUDA if available."""


def main() -> None:
    args = tyro.cli(EvalArgs)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    print(f"Using device: {device}")

    print("Initializing Custom_enviornments.Test_Env.Env_16.Env16...")
    env = Env16()
    try:
        print(f"Loading model from {args.model_path}...")
        model = Agent(env).to(device)
        model.load_state_dict(
            torch.load(
                args.model_path,
                map_location=device,
                weights_only=True,
            )
        )

        print(f"Starting evaluation for {args.eval_episodes} episodes...")
        start_time = time.time()
        episodic_returns, wins = evaluate(
            env,
            model,
            eval_episodes=args.eval_episodes,
            device=device,
            deterministic=args.deterministic,
        )
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 50)
        print(f"Evaluation Results ({args.eval_episodes} episodes)")
        print("=" * 50)
        print(f"Mean Return: {np.mean(episodic_returns):.2f}")
        print(f"Std Return: {np.std(episodic_returns):.2f}")
        print(f"Max Return: {np.max(episodic_returns):.2f}")
        print(f"Min Return: {np.min(episodic_returns):.2f}")
        print(f"Wins: {wins}/{args.eval_episodes}")
        print(f"Win Rate: {100.0 * wins / args.eval_episodes:.2f}%")
        print(f"Total Time: {elapsed_time:.2f}s")
        print("=" * 50)
    finally:
        env.close()


if __name__ == "__main__":
    main()
