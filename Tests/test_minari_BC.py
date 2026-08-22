"""Sample episodes from a local Minari behavior-cloning dataset."""

from __future__ import annotations

import argparse
import json
from typing import Any

import minari


DEFAULT_DATASET_ID = "env16/BC-v0"
DEFAULT_EPISODES = 5


def _json_value(value: Any) -> str:
    """Return NumPy-backed Minari values as readable JSON."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def print_sampled_episodes(
    dataset_id: str = DEFAULT_DATASET_ID,
    number_of_episodes: int = DEFAULT_EPISODES,
) -> None:
    """Load a dataset and print every transition in sampled episodes."""

    if number_of_episodes <= 0:
        raise ValueError("number_of_episodes must be greater than zero.")

    dataset = minari.load_dataset(dataset_id)
    sample_size = min(number_of_episodes, dataset.total_episodes)
    if sample_size == 0:
        raise ValueError(f"Minari dataset {dataset_id!r} contains no episodes.")

    episodes = dataset.sample_episodes(n_episodes=sample_size)
    for sample_index, episode_data in enumerate(episodes, start=1):
        print(
            f"episode_sample={sample_index} episode_id={episode_data.id} "
            f"transitions={len(episode_data.actions)}"
        )
        for step, action in enumerate(episode_data.actions):
            print(
                f"  step={step} "
                f"obs={_json_value(episode_data.observations[step])} "
                f"rew={float(episode_data.rewards[step])} "
                f"action={_json_value(action)} "
                f"terminated={bool(episode_data.terminations[step])} "
                f"truncated={bool(episode_data.truncations[step])}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and print transitions from a local Minari dataset."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of episodes to sample (default: {DEFAULT_EPISODES}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print_sampled_episodes(args.dataset_id, args.episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
