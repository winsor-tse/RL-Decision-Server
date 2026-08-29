"""Record live human demonstrations as a Minari Env16BC dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import minari
from minari.dataset.minari_dataset import parse_dataset_id
from minari.storage.datasets_root_dir import get_dataset_path

from Custom_enviornments.Test_Env.Env_16_BC import Env16BC
from Offline.record_player import parse_no_op_action


DEFAULT_DATASET_ID = "env16/BC-v0"
DEFAULT_MAX_STEPS = 500


@contextmanager
def _defer_sigint():
    """Defer Ctrl+C while Minari is mutating its on-disk dataset."""

    state = {"received": False}
    if threading.current_thread() is not threading.main_thread():
        yield state
        return

    previous_handler = signal.getsignal(signal.SIGINT)

    def remember_interrupt(_signum, _frame):
        state["received"] = True

    signal.signal(signal.SIGINT, remember_interrupt)
    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _save_collector(
    *,
    collector: minari.DataCollector,
    dataset,
    dataset_id: str,
    author: str,
    reason: str,
):
    """Persist the collector's completed data and return a refreshed dataset."""

    if dataset is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`(code_permalink|author_email)` is set to None.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"`eval_env` is set to None.*",
            )
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
    else:
        collector.add_to_dataset(dataset)
        # Minari caches total_steps on loaded dataset objects. Reload after an
        # append so the printed counts reflect what is actually on disk.
        dataset = minari.load_dataset(dataset_id)

    print(
        f"dataset_saved=True reason={reason} id={dataset.id} "
        f"episodes={dataset.total_episodes} transitions={dataset.total_steps} "
        f"path={dataset.spec.data_path}",
        flush=True,
    )
    return dataset


def _save_collector_safely(**kwargs):
    with _defer_sigint() as interrupt_state:
        dataset = _save_collector(**kwargs)
    return dataset, bool(interrupt_state["received"])


def _observation_text(observation: Any) -> str:
    value = observation.tolist() if hasattr(observation, "tolist") else observation
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _collect_and_save(
    *,
    collector: minari.DataCollector,
    base_env: Env16BC,
    dataset_id: str,
    max_steps: int,
    author: str,
):
    recorded_steps = 0
    unsaved_steps = 0
    dataset = None
    interrupted = False

    try:
        collector.reset(options={"minari_autoseed": False})

        while recorded_steps < max_steps:
            # The action was captured from the current observation. step()
            # waits for the next labeled frame, preserving Minari's
            # (observation_t, action_t, observation_t+1) alignment.
            action = base_env.next_action()
            observation, _, terminated, truncated, _ = collector.step(action)
            recorded_steps += 1
            unsaved_steps += 1
            print(
                f"key={base_env.last_recorded_key} "
                f"state={_observation_text(observation)} "
                f"terminated={bool(terminated)} truncated={bool(truncated)}",
                flush=True,
            )

            if terminated or truncated:
                if terminated and truncated:
                    reason = "terminated+truncated"
                elif terminated:
                    reason = "terminated"
                else:
                    reason = "truncated"
                # Mark this batch claimed before the non-transactional Minari
                # write. Never blindly retry an interrupted create/append.
                unsaved_steps = 0
                dataset, save_was_interrupted = _save_collector_safely(
                    collector=collector,
                    dataset=dataset,
                    dataset_id=dataset_id,
                    author=author,
                    reason=reason,
                )
                if save_was_interrupted:
                    raise KeyboardInterrupt
                if recorded_steps < max_steps:
                    collector.reset(options={"minari_autoseed": False})
    except KeyboardInterrupt:
        interrupted = True

    if unsaved_steps > 0:
        dataset, _ = _save_collector_safely(
            collector=collector,
            dataset=dataset,
            dataset_id=dataset_id,
            author=author,
            reason="interrupt" if interrupted else "max_steps",
        )
    elif dataset is None:
        print(
            "dataset_saved=False reason=no_complete_transitions",
            flush=True,
        )

    return dataset


def record_minari_dataset(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    max_steps: int = DEFAULT_MAX_STEPS,
    no_op_action: Any = "NoOp",
    author: str = "Yugen Saga player",
    raw_env: Env16BC | None = None,
):
    """Collect at most ``max_steps`` valid human actions and save a dataset.

    Invalid-key ticks are serviced inside :class:`Env16BC` and never become
    actions. Keyless death/truncation ticks may complete the preceding valid
    action. Every episode boundary is persisted immediately, and a partial
    final episode is flushed as truncated when collection stops or Ctrl+C is
    pressed.
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
    try:
        return _collect_and_save(
            collector=collector,
            base_env=base_env,
            dataset_id=dataset_id,
            max_steps=max_steps,
            author=author,
        )
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
        level=logging.WARNING,
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
