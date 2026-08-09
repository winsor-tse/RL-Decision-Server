import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from torch.distributions.categorical import Categorical

from Custom_enviornments.Test_Env.Env_16 import Env16
from Training.PPO_lstm_server import Agent
from Utils.model_paths import inference_checkpoint_path

LSTMState = tuple[torch.Tensor, torch.Tensor]


def initial_lstm_state(model: Agent, device: torch.device) -> LSTMState:
    """Create the empty recurrent state used at an episode boundary."""
    shape = (model.lstm.num_layers, 1, model.lstm.hidden_size)
    return (
        torch.zeros(shape, device=device),
        torch.zeros(shape, device=device),
    )


def select_action(
    model: Agent,
    observation: np.ndarray,
    lstm_state: LSTMState,
    device: torch.device,
    deterministic: bool,
) -> tuple[int, LSTMState]:
    """Select one action and advance the policy's recurrent state."""
    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    episode_start = torch.zeros(1, dtype=torch.float32, device=device)

    with torch.no_grad():
        embedding, next_lstm_state = model.get_states(
            observation_tensor,
            lstm_state,
            episode_start,
        )
        logits = model.actor(embedding)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = Categorical(logits=logits).sample()

    return int(action.item()), next_lstm_state


def evaluate(
    env,
    model: Agent,
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
    deterministic: bool = True,
) -> tuple[list[float], int]:
    """Evaluate recurrent PPO directly against the live Env16 stream."""
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be greater than zero")

    model.eval()
    episodic_returns = []
    wins = 0

    for episode in range(eval_episodes):
        observation, _ = env.reset()
        lstm_state = initial_lstm_state(model, device)
        episode_return = 0.0
        done = False
        info = {}

        while not done:
            action, lstm_state = select_action(
                model,
                observation,
                lstm_state,
                device,
                deterministic,
            )
            observation, reward, terminated, truncated, info = env.step(
                np.array([action])
            )
            episode_return += float(reward)
            done = bool(terminated or truncated)

        episode_won = bool(info.get("is_win", False))
        episodic_returns.append(episode_return)
        wins += int(episode_won)
        outcome = "win" if episode_won else info.get(
            "episode_outcome",
            "loss",
        )
        print(
            f"eval_episode={episode}, episodic_return={episode_return}, "
            f"outcome={outcome}"
        )

    return episodic_returns, wins


@dataclass
class EvalArgs:
    model_path: str | None = None
    """Checkpoint override; defaults to the newest PPO-LSTM run checkpoint."""
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
    model_path = inference_checkpoint_path(
        args.model_path,
        "PPO_lstm_server.pt",
    )

    env = Env16()
    try:
        print(f"Loading recurrent PPO model from {model_path}...")
        model = Agent(env).to(device)
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
        )

        start_time = time.time()
        episodic_returns, wins = evaluate(
            env,
            model,
            eval_episodes=args.eval_episodes,
            device=device,
            deterministic=args.deterministic,
        )
        elapsed_time = time.time() - start_time

        print(f"Mean Return: {np.mean(episodic_returns):.2f}")
        print(f"Std Return: {np.std(episodic_returns):.2f}")
        print(f"Wins: {wins}/{args.eval_episodes}")
        print(f"Win Rate: {100.0 * wins / args.eval_episodes:.2f}%")
        print(f"Total Time: {elapsed_time:.2f}s")
    finally:
        env.close()


if __name__ == "__main__":
    main()
