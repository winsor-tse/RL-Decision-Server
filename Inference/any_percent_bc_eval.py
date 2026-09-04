"""Evaluate an any-percent behavior-cloning checkpoint.

Dataset mode compares checkpoint predictions with recorded Minari actions.
Live mode runs the same policy against ``Env16`` and is intended to be started
through ``RunOfflineRL.ps1`` so the WebSocket bridge is supervised alongside
the evaluator.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import minari
import numpy as np
import torch
import torch.nn as nn

from Custom_enviornments.Test_Env.Env_16 import Env16
from Custom_enviornments.Test_Env.Env_16_BC import BC_ACTIONS_11


DEFAULT_DATASET_ID = "env16/BC-v0"


class Actor(nn.Module):
    """Network architecture used by ``Offline.any_percent_bc`` checkpoints."""

    def __init__(self, state_dim: int, max_action: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


@dataclass(frozen=True)
class PreparedDataset:
    observations: np.ndarray
    actions: np.ndarray
    state_mean: np.ndarray | float
    state_std: np.ndarray | float
    max_action: float


def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """Compute the discounted return used to rank BC demonstrations."""

    total = 0.0
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
    return total


def prepare_dataset(
    dataset_id: str,
    *,
    top_fraction: float,
    gamma: float,
    normalize_state: bool,
) -> PreparedDataset:
    """Load and prepare the same top trajectory fraction used for training."""

    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be greater than 0 and at most 1.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    dataset = minari.load_dataset(dataset_id)
    ranked_episodes = sorted(
        dataset.iterate_episodes(),
        key=lambda episode: discounted_return(episode.rewards, gamma),
        reverse=True,
    )
    sample_count = max(1, int(top_fraction * len(ranked_episodes)))
    selected_episodes = ranked_episodes[:sample_count]
    if not selected_episodes:
        raise ValueError(f"Minari dataset {dataset_id!r} contains no episodes.")

    observations = np.concatenate(
        [episode.observations[:-1] for episode in selected_episodes]
    ).astype(np.float32)
    actions = np.concatenate(
        [episode.actions for episode in selected_episodes]
    ).astype(np.float32).reshape(-1)
    if observations.size == 0 or actions.size == 0:
        raise ValueError(f"Minari dataset {dataset_id!r} contains no transitions.")

    if normalize_state:
        state_mean = observations.mean(axis=0)
        state_std = observations.std(axis=0) + 1e-3
        observations = (observations - state_mean) / state_std
    else:
        state_mean = 0.0
        state_std = 1.0

    max_action = max(1.0, float(np.max(np.abs(actions))))
    return PreparedDataset(
        observations=observations,
        actions=actions,
        state_mean=state_mean,
        state_std=state_std,
        max_action=max_action,
    )


def load_actor(
    checkpoint_path: str | Path,
    prepared: PreparedDataset,
    device: torch.device,
) -> Actor:
    """Construct the BC actor and load its checkpoint weights."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint does not exist: {checkpoint}")

    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    actor_state = state.get("actor", state) if isinstance(state, dict) else state
    actor = Actor(
        state_dim=int(prepared.observations.shape[1]),
        max_action=prepared.max_action,
    ).to(device)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


@torch.no_grad()
def predict_actions(
    actor: Actor,
    observations: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    observations_tensor = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=device,
    )
    return actor(observations_tensor).cpu().numpy().reshape(-1)


def discrete_bc_actions(predictions: np.ndarray) -> np.ndarray:
    """Round scalar BC outputs to valid Env16BC action indices."""

    return np.clip(
        np.rint(predictions),
        0,
        len(BC_ACTIONS_11) - 1,
    ).astype(np.int64)


def save_predictions(
    output_path: str | Path,
    predictions: np.ndarray,
    predicted_actions: np.ndarray,
    recorded_actions: np.ndarray,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["predicted_value", "predicted_action", "recorded_action"]
        )
        writer.writerows(
            zip(predictions, predicted_actions, recorded_actions, strict=True)
        )
    return destination


def evaluate_dataset(
    actor: Actor,
    prepared: PreparedDataset,
    device: torch.device,
    output_csv: str | Path,
) -> tuple[float, float]:
    """Evaluate prediction error on recorded transitions and write a CSV."""

    predictions = predict_actions(actor, prepared.observations, device)
    predicted_actions = discrete_bc_actions(predictions)
    recorded_actions = prepared.actions.astype(np.int64)
    mse = float(np.mean(np.square(predictions - prepared.actions)))
    accuracy = float(np.mean(predicted_actions == recorded_actions))
    destination = save_predictions(
        output_csv,
        predictions,
        predicted_actions,
        recorded_actions,
    )

    print(f"transitions={len(recorded_actions)}")
    print(f"action_mse={mse:.6f}")
    print(f"discrete_action_accuracy={accuracy:.2%}")
    print(f"predictions_saved={destination}")
    return mse, accuracy


def env16_action_from_bc_index(env: Env16, bc_action: int) -> int:
    """Translate Env16BC's action ordering into Env16's action ordering."""

    action_name = BC_ACTIONS_11[bc_action]
    return env.Actions.index(action_name)


def evaluate_live(
    env: Env16,
    actor: Actor,
    prepared: PreparedDataset,
    *,
    eval_episodes: int,
    device: torch.device,
) -> tuple[list[float], int]:
    """Run a BC checkpoint against the live Env16 game stream."""

    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be greater than zero.")

    episode_returns = []
    wins = 0
    for episode in range(eval_episodes):
        observation, _ = env.reset(seed=episode)
        episode_return = 0.0
        done = False
        info = {}
        while not done:
            normalized = (
                np.asarray(observation, dtype=np.float32) - prepared.state_mean
            ) / prepared.state_std
            prediction = predict_actions(
                actor,
                normalized.reshape(1, -1),
                device,
            )
            bc_action = int(discrete_bc_actions(prediction)[0])
            env_action = env16_action_from_bc_index(env, bc_action)
            observation, reward, terminated, truncated, info = env.step(
                env_action
            )
            episode_return += float(reward)
            done = bool(terminated or truncated)

        won = bool(info.get("is_win", False))
        wins += int(won)
        episode_returns.append(episode_return)
        outcome = info.get("episode_outcome") or (
            "win" if won else "loss"
        )
        print(
            f"eval_episode={episode} return={episode_return:.2f} "
            f"outcome={outcome}"
        )

    print(f"mean_return={np.mean(episode_returns):.2f}")
    print(f"wins={wins}/{eval_episodes}")
    print(f"win_rate={100.0 * wins / eval_episodes:.2f}%")
    return episode_returns, wins


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return torch.device(normalized)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--mode", choices=("dataset", "live"), default="dataset")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--top-fraction", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--normalize-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-csv",
        help="Dataset-mode CSV path; defaults beside the checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    prepared = prepare_dataset(
        args.dataset_id,
        top_fraction=args.top_fraction,
        gamma=args.gamma,
        normalize_state=args.normalize_state,
    )
    actor = load_actor(args.checkpoint_path, prepared, device)
    print(f"device={device}")
    print(f"checkpoint={Path(args.checkpoint_path).expanduser().resolve()}")
    print(f"dataset={args.dataset_id}")

    if args.mode == "dataset":
        output_csv = args.output_csv or str(
            Path(args.checkpoint_path)
            .expanduser()
            .resolve()
            .with_name("predicted_actions.csv")
        )
        evaluate_dataset(actor, prepared, device, output_csv)
        return 0

    env = Env16()
    try:
        evaluate_live(
            env,
            actor,
            prepared,
            eval_episodes=args.eval_episodes,
            device=device,
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
