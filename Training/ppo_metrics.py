from collections.abc import Sequence
from typing import Protocol

import numpy as np


class TensorBoardWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value, global_step: int) -> None: ...

    def add_histogram(
        self,
        tag: str,
        values,
        global_step: int,
        bins,
    ) -> None: ...


def log_step_metrics(
    writer: TensorBoardWriter,
    *,
    global_step: int,
    reward: float,
    observation: np.ndarray,
    player_hp_index: int,
    enemy_hp_index: int,
    reward_components: dict[str, float],
    action_counts: np.ndarray,
    action_names: Sequence[str],
    recent_actions: Sequence[int],
) -> None:
    """Write the environment dashboard shared by PPO trainers."""
    writer.add_scalar("charts/step_reward", reward, global_step)
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
