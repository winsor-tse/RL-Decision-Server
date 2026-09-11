"""Versioned, atomic training checkpoints shared by PPO trainers."""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


CHECKPOINT_VERSION = 1

# Invocation controls and computed values are intentionally not restored.
_CONTROL_FIELDS = {
    "resume_checkpoint_path",
    "restore_model_path",
    "model_path",
    "training_checkpoint_path",
    "checkpoint_interval",
    "stop_after_timesteps",
    "batch_size",
    "minibatch_size",
    "num_iterations",
}


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            torch.as_tensor(numpy_state["keys"]).cpu().numpy().astype(
                np.uint32, copy=False
            ),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.as_tensor(state["torch"]).cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value).cpu() for value in state["cuda"]]
        )


def atomic_torch_save(payload: Any, checkpoint_path: str | Path) -> Path:
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def save_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    algorithm: str,
    args: Any,
    agent: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    completed_iteration: int,
    run_name: str,
    run_directory: str | Path,
    runtime_state: dict[str, Any],
    recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "algorithm": algorithm,
        "args": asdict(args),
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "completed_iteration": int(completed_iteration),
        "run_name": run_name,
        "run_directory": str(run_directory),
        "model_path": str(args.model_path),
        "runtime_state": runtime_state,
        "rng_state": capture_rng_state(),
        "metadata": {
            "saved_at_unix": time.time(),
            "torch_version": str(torch.__version__),
            **(metadata or {}),
        },
    }
    if recurrent_state is not None:
        payload["recurrent_state"] = tuple(
            value.detach().cpu() for value in recurrent_state
        )
    return atomic_torch_save(payload, checkpoint_path)


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    algorithm: str,
    device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Not a supported PPO training checkpoint: {path}")
    if checkpoint.get("algorithm") != algorithm:
        raise ValueError(
            f"Checkpoint is for {checkpoint.get('algorithm')!r}, not {algorithm!r}"
        )
    required = {
        "args", "agent", "optimizer", "global_step", "completed_iteration",
        "run_name", "run_directory", "runtime_state", "rng_state",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Training checkpoint is missing: {', '.join(missing)}")
    return path, checkpoint


def restore_saved_hyperparameters(args: Any, checkpoint: dict[str, Any]) -> None:
    """Use the original run's hyperparameters while retaining invocation controls."""

    saved = checkpoint["args"]
    for field, value in saved.items():
        if field not in _CONTROL_FIELDS and hasattr(args, field):
            setattr(args, field, value)
