import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Training.DQN_server import QNetwork
from Custom_enviornments.Test_Env import Env_16

"""
# Basic usage with defaults
python DQN_eval.py --model_path runs/DQN_server__1234567890/DQN_server.pt

# Custom eval episodes and epsilon
python DQN_eval.py --model_path runs/DQN_server__1234567890/DQN_server.pt --eval_episodes 20 --epsilon 0.05

# Force CPU
python DQN_eval.py --model_path runs/DQN_server__1234567890/DQN_server.pt --cuda false
"""

def evaluate(
    env,
    model: torch.nn.Module,
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.0,
) -> tuple[list[float], int]:
    """Evaluate a trained DQN model on a single custom environment."""
    model.eval()
    episodic_returns = []
    wins = 0

    for episode in range(eval_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        episode_won = False
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.single_action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(torch.Tensor(obs).to(device))
                    action = torch.argmax(q_values).cpu().item()

            next_obs, reward, terminated, truncated, info = env.step(np.array([action]))
            episode_return += float(reward)
            done = bool(terminated or truncated)
            if done:
                episode_won = bool(info.get("is_win", False))
            obs = next_obs

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
    """Path to the trained PyTorch model (.pt file)."""
    eval_episodes: int = 10
    """Number of evaluation episodes to run."""
    epsilon: float = 0.0
    """Exploration rate during evaluation (0.0 for pure greedy)."""
    cuda: bool = True
    """Whether to use CUDA if available."""


if __name__ == "__main__":
    args = tyro.cli(EvalArgs)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    print("Initializing Custom_enviornments.Test_Env.Env_16.Env16...")
    env = Env_16.Env16()

    print(f"Loading model from {args.model_path}...")
    model = QNetwork(env).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print(f"Starting evaluation for {args.eval_episodes} episodes...")
    start_time = time.time()
    episodic_returns, wins = evaluate(
        env,
        model,
        eval_episodes=args.eval_episodes,
        device=device,
        epsilon=args.epsilon,
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

    env.close()
