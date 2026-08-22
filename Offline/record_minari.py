"""Record live human demonstrations as a Minari Env16BC dataset."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import minari
from minari.dataset.minari_dataset import parse_dataset_id
from minari.storage.datasets_root_dir import get_dataset_path

from Custom_enviornments.Test_Env.Env_16_BC import Env16BC
from Offline.record_player import parse_no_op_action


LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_ID = "env16/BC-v0"
DEFAULT_MAX_STEPS = 500


def record_minari_dataset(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    max_steps: int = DEFAULT_MAX_STEPS,
    no_op_action: Any = "NoOp",
    author: str = "Yugen Saga player",
    raw_env: Env16BC | None = None,
):
    """Collect at most ``max_steps`` valid human actions and save a dataset.

    Invalid-key ticks are serviced inside :class:`Env16BC` and never reach the
    ``DataCollector``. A partial final episode is intentionally flushed by
    Minari as truncated when collection stops or Ctrl+C is pressed.
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")
    # Validate before opening the live socket and waiting for the game.
    parse_dataset_id(dataset_id)
    destination = get_dataset_path(dataset_id)
    if destination.exists():
        raise FileExistsError(
            f"Minari dataset {dataset_id!r} already exists at {destination}. "
            "Choose a new version, for example env16/BC-v1."
        )

    base_env = (
        raw_env if raw_env is not None else Env16BC(no_op_action=no_op_action)
    )
    collector = minari.DataCollector(base_env, record_infos=True)
    recorded_steps = 0
    completed_episodes = 0
    current_return = 0.0
    interrupted = False

    try:
        print("Waiting for the first valid human action...", flush=True)
        collector.reset(options={"minari_autoseed": False})

        try:
            while recorded_steps < max_steps:
                # The action was captured from the current observation. step()
                # waits for the next valid labeled frame, preserving Minari's
                # (observation_t, action_t, observation_t+1) alignment.
                action = base_env.next_action()
                _, reward, terminated, truncated, info = collector.step(action)
                recorded_steps += 1
                current_return += float(reward)
                print(
                    f"[RECORDED] step={recorded_steps}/{max_steps} "
                    f"key={base_env.last_recorded_key} action={action} "
                    f"label={base_env.Actions[action]} reward={float(reward):.4f}",
                    flush=True,
                )

                if terminated or truncated:
                    completed_episodes += 1
                    print(
                        f"[EPISODE END] episode={completed_episodes} "
                        f"return={current_return:.4f} "
                        f"outcome={info['episode_outcome']}",
                        flush=True,
                    )
                    current_return = 0.0
                    if recorded_steps < max_steps:
                        print("Waiting for the next episode...", flush=True)
                        collector.reset(options={"minari_autoseed": False})
        except KeyboardInterrupt:
            interrupted = True
            print("\nStopping collection and flushing recorded steps...", flush=True)

        if recorded_steps == 0:
            print("No complete transitions were recorded; no dataset was created.")
            return None

        dataset = collector.create_dataset(
            dataset_id=dataset_id,
            algorithm_name="human behavior cloning demonstrations",
            author=author,
            description=(
                "Human Yugen Saga demonstrations recorded with Env16BC. "
                "The game received NoOp responses; only mapped keyboard "
                "actions were included."
            ),
            requirements=["minari==0.5.3"],
        )
        print(
            f"Created {dataset.id!r}: {dataset.total_episodes} episodes, "
            f"{dataset.total_steps} transitions at {dataset.spec.data_path}",
            flush=True,
        )
        if interrupted:
            LOGGER.info("Dataset creation followed a keyboard interrupt.")
        return dataset
    finally:
        collector.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record valid human Env16 actions into a Minari dataset."
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Versioned Minari ID (default: {DEFAULT_DATASET_ID})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum valid transitions to record (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--no-op-action",
        type=parse_no_op_action,
        default="NoOp",
        help='JSON value sent to the game for every tick (default: "NoOp")',
    )
    parser.add_argument(
        "--author",
        default="Yugen Saga player",
        help="Author stored in the Minari metadata.",
    )
    parser.add_argument(
        "--datasets-path",
        type=Path,
        help="Override MINARI_DATASETS_PATH for this recording.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.datasets_path is not None:
        datasets_path = args.datasets_path.expanduser().resolve()
        datasets_path.mkdir(parents=True, exist_ok=True)
        os.environ["MINARI_DATASETS_PATH"] = str(datasets_path)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    record_minari_dataset(
        dataset_id=args.dataset_id,
        max_steps=args.max_steps,
        no_op_action=args.no_op_action,
        author=args.author,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
