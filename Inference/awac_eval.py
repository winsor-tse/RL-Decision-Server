"""Evaluate a discrete AWAC checkpoint against the live Env16 game."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from Custom_enviornments.Test_Env.Env_16 import Env16
from Custom_enviornments.Test_Env.Env_16_BC import BC_ACTIONS_11
from Offline.awac import AWAC_MODEL_FILENAME, Actor
from Utils.model_paths import inference_checkpoint_path


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return torch.device(normalized)


def load_actor(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Actor, np.ndarray, np.ndarray, list[str]]:
    """Restore the actor and preprocessing metadata saved during AWAC training."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AWAC checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {
        "actor", "state_mean", "state_std", "state_dim", "action_dim",
        "hidden_dim", "actions",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            f"AWAC checkpoint is missing required fields: {', '.join(missing)}"
        )

    actions = list(checkpoint["actions"])
    if actions != list(BC_ACTIONS_11):
        raise ValueError("AWAC checkpoint action ordering does not match Env16 BC actions")
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    if action_dim != len(actions):
        raise ValueError("AWAC checkpoint action_dim does not match its action metadata")

    state_mean = torch.as_tensor(checkpoint["state_mean"]).cpu().numpy().astype(
        np.float32, copy=False
    ).reshape(-1)
    state_std = torch.as_tensor(checkpoint["state_std"]).cpu().numpy().astype(
        np.float32, copy=False
    ).reshape(-1)
    if state_mean.size != state_dim or state_std.size != state_dim:
        raise ValueError("AWAC checkpoint normalization shape does not match state_dim")
    if not np.all(np.isfinite(state_mean)) or not np.all(np.isfinite(state_std)):
        raise ValueError("AWAC checkpoint normalization contains non-finite values")
    if np.any(state_std <= 0):
        raise ValueError("AWAC checkpoint state_std must be positive")

    actor = Actor(state_dim, action_dim, int(checkpoint["hidden_dim"])).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor, state_mean, state_std, actions


@torch.no_grad()
def select_action(
    actor: Actor,
    observation: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
    *,
    deterministic: bool,
) -> int:
    normalized = (np.asarray(observation, dtype=np.float32) - state_mean) / state_std
    state = torch.as_tensor(normalized, dtype=torch.float32, device=device).reshape(1, -1)
    policy = actor._get_policy(state)
    action = policy.probs.argmax(-1) if deterministic else policy.sample()
    return int(action.item())


def evaluate_live(
    env: Env16,
    actor: Actor,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    *,
    eval_episodes: int,
    device: torch.device,
    deterministic: bool = True,
) -> tuple[list[float], int]:
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be greater than zero")

    returns: list[float] = []
    wins = 0
    for episode in range(eval_episodes):
        observation, _ = env.reset(seed=episode)
        episode_return = 0.0
        done = False
        info = {}
        while not done:
            action = select_action(
                actor, observation, state_mean, state_std, device,
                deterministic=deterministic,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = bool(terminated or truncated)

        won = bool(info.get("is_win", False))
        wins += int(won)
        returns.append(episode_return)
        outcome = info.get("episode_outcome") or ("win" if won else "loss")
        print(
            f"eval_episode={episode} return={episode_return:.2f} "
            f"outcome={outcome}"
        )

    print(f"mean_return={np.mean(returns):.2f}")
    print(f"std_return={np.std(returns):.2f}")
    print(f"wins={wins}/{eval_episodes}")
    print(f"win_rate={100.0 * wins / eval_episodes:.2f}%")
    return returns, wins


@dataclass
class EvalArgs:
    checkpoint_path: str | None = None
    """Checkpoint override; defaults to the newest AWAC_model.pt under runs."""
    eval_episodes: int = 5
    """Number of live episodes."""
    device: str = "auto"
    """Evaluation device: auto, cpu, or cuda."""
    deterministic: bool = True
    """Choose the highest-probability action; disable to sample the policy."""


def main() -> None:
    args = tyro.cli(EvalArgs)
    device = resolve_device(args.device)
    checkpoint_path = inference_checkpoint_path(
        args.checkpoint_path,
        AWAC_MODEL_FILENAME,
    )
    actor, state_mean, state_std, actions = load_actor(checkpoint_path, device)
    print(f"device={device}")
    print(f"checkpoint={checkpoint_path}")
    print("Initializing Env16; waiting for the first game ai_tick...")
    env = Env16(actions=actions)
    try:
        evaluate_live(
            env,
            actor,
            state_mean,
            state_std,
            eval_episodes=args.eval_episodes,
            device=device,
            deterministic=args.deterministic,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
