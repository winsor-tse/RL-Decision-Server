"""Human-demonstration variant of :mod:`Env_16` for Minari collection.

``Env16BC`` never sends the captured player action back to the game. Every
ZeroMQ tick receives a ``NoOp`` response; valid keyboard actions are only used
as labels in the offline dataset. Ticks without a mapped key are acknowledged
but are not exposed as Gym steps, except when they finish the preceding valid
action with a terminal or truncated state.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gymnasium.envs.registration import EnvSpec

from Custom_enviornments.BaseEnv import BaseEnv
from Custom_enviornments.Test_Env import Env_conditions
from Custom_enviornments.Test_Env.Env_16 import Env16
from Offline.record_player import WindowsInputCapture


LOGGER = logging.getLogger(__name__)
ENV16_BC_ENV_ID = "YugenSaga/Env16BC-v0"

# This order intentionally follows the indices requested for the BC dataset.
# It is separate from Env_16.ACTIONS_11, whose movement order is
# [up, down, left, right].
BC_ACTIONS_11 = [
    "up",
    "left",
    "right",
    "down",
    "attack",
    "castSpell:1",
    "castSpell:2",
    "castSpell:3",
    "castSpell:5",
    "castSpell:6",
    "castSpell:7",
]

KEY_TO_ACTION_INDEX = {
    "W": 0,
    "A": 1,
    "D": 2,
    "S": 3,
    "SPACE": 4,
    "1": 5,
    "2": 6,
    "3": 7,
    "5": 8,
    "6": 9,
    "7": 10,
    "NUMPAD1": 5,
    "NUMPAD2": 6,
    "NUMPAD3": 7,
    "NUMPAD5": 8,
    "NUMPAD6": 9,
    "NUMPAD7": 10,
}

REWARD_COMPONENT_KEYS = (
    "health_state",
    "positioning",
    "damage_taken",
    "damage_dealt",
    "terminal",
    "killed",
)


def _normalize_key_name(key: Any) -> str:
    key_name = str(key)
    return "SPACE" if key_name == " " else key_name.strip().upper()


def action_from_input(player_input: dict[str, Any]) -> tuple[int, str] | None:
    """Return the mapped ``(action_index, key)`` from an input snapshot.

    New key-down events take precedence over held keys. This makes a newly
    pressed attack/spell win over a movement key that is still held. Two new
    mapped actions, or two mapped held actions without a new event, are
    ambiguous for a Discrete action space and are ignored. Top-row and numpad
    aliases for the same spell count as one action. Unmapped keys do not make
    an otherwise unambiguous action invalid.
    """

    key_down_events = [
        _normalize_key_name(event.get("key", ""))
        for event in player_input.get("events", [])
        if event.get("event") == "key_down"
    ]
    mapped_events = list(
        dict.fromkeys(key for key in key_down_events if key in KEY_TO_ACTION_INDEX)
    )
    mapped_event_actions = {
        KEY_TO_ACTION_INDEX[key] for key in mapped_events
    }
    if len(mapped_event_actions) == 1:
        key = mapped_events[0]
        return KEY_TO_ACTION_INDEX[key], key
    if len(mapped_event_actions) > 1:
        return None

    held_keys = [
        _normalize_key_name(held_key.get("key", ""))
        for held_key in player_input.get("keys_down", [])
    ]
    mapped_held_keys = list(
        dict.fromkeys(key for key in held_keys if key in KEY_TO_ACTION_INDEX)
    )
    mapped_held_actions = {
        KEY_TO_ACTION_INDEX[key] for key in mapped_held_keys
    }
    if len(mapped_held_actions) == 1:
        key = mapped_held_keys[0]
        return KEY_TO_ACTION_INDEX[key], key

    return None


class Env16BC(Env16):
    """Env16 reward/state logic with human actions and no-op game replies."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        input_capture=None,
        no_op_action: Any = "NoOp",
        config=None,
        socket=None,
    ):
        super().__init__(actions=BC_ACTIONS_11, config=config, socket=socket)
        # Directly constructed custom envs normally have spec=None. Providing
        # one lets Minari retain enough metadata to recover this environment.
        self.spec = EnvSpec(
            id=ENV16_BC_ENV_ID,
            entry_point=(
                "Custom_enviornments.Test_Env.Env_16_BC:Env16BC"
            ),
            max_episode_steps=int(self.config["MAX_EPISODE_STEPS"]),
        )
        self.input_capture = (
            input_capture if input_capture is not None else WindowsInputCapture()
        )
        self.no_op_action = no_op_action
        self.ignored_ticks = 0
        self._pending_action: int | None = None
        self._pending_key = ""
        self.last_recorded_key = ""
        self.last_recorded_action: int | None = None

    @staticmethod
    def _invalid_message_response(message: Any) -> dict[str, Any]:
        request_id = message.get("requestId") if isinstance(message, dict) else None
        return {
            "type": "error",
            "requestId": request_id,
            "error": "ZeroMQ payload must be an ai_tick JSON object.",
        }

    def _receive_valid_action_tick(
        self,
        *,
        reset: bool,
        accept_episode_boundary: bool = False,
    ) -> tuple[dict[str, Any], int | None, str, np.ndarray | None]:
        """Wait for a mapped action or an allowed keyless episode boundary."""

        reset_response_pending = reset
        while True:
            message = self.socket.recv_json()
            if not isinstance(message, dict) or message.get("type") != "ai_tick":
                self.socket.send_json(self._invalid_message_response(message))
                LOGGER.warning("Ignored non-ai_tick payload: %r", message)
                continue

            player_input = self.input_capture.snapshot()
            mapped_action = action_from_input(player_input)
            response = self._build_response(
                message=message,
                move=self.no_op_action,
                reset=reset_response_pending,
            )
            self.socket.send_json(response)
            reset_response_pending = False

            if mapped_action is None:
                if accept_episode_boundary:
                    candidate_state = Env_conditions.parse_observation(
                        message.get("worldState", {}),
                        int(self.config["OBS_SIZE"]),
                    )
                    is_loss = Env_conditions.is_episode_loss(
                        candidate_state,
                        self.next_state,
                    )
                    is_truncated = Env_conditions.get_truncated(
                        candidate_state,
                        self.next_state,
                        self.current_step + 1,
                    )
                    if is_loss or is_truncated:
                        return message, None, "", candidate_state

                self.ignored_ticks += 1
                LOGGER.debug(
                    "Ignored tick %r without a mapped key.",
                    message.get("requestId"),
                )
                continue

            action_idx, key = mapped_action
            return message, action_idx, key, None

    @staticmethod
    def _minari_info(info: dict[str, Any]) -> dict[str, Any]:
        """Return a fixed, HDF5-safe info tree for every episode item."""

        reward_components = info.get("reward_components", {})
        return {
            "current_step": int(info["current_step"]),
            "next_state": info["next_state"],
            "reward_components": {
                key: float(reward_components.get(key, 0.0))
                for key in REWARD_COMPONENT_KEYS
            },
            "is_win": bool(info.get("is_win", False)),
            "episode_outcome": str(info.get("episode_outcome") or "ongoing"),
        }

    def reset(self, seed=None, options=None):
        # Bypass Env16.reset because it sends direction:up and does not capture
        # a human action for the returned initial observation.
        BaseEnv.reset(self, seed=seed, options=options)
        self.input_capture.reset()
        self._pending_action = None
        self._pending_key = ""

        message, action_idx, key, _ = self._receive_valid_action_tick(reset=True)
        assert action_idx is not None
        observation, info = self._initialize_from_world_state(
            message.get("worldState", {})
        )
        self._pending_action = action_idx
        self._pending_key = key
        return observation, self._minari_info(info)

    def next_action(self) -> int:
        """Return the valid human action captured for the current observation."""

        if self._pending_action is None:
            raise RuntimeError(
                "No captured action is ready. Call reset() after an episode ends."
            )
        return self._pending_action

    def step(self, action):
        action_idx = self._normalize_action(action)
        if action_idx < 0 or action_idx >= len(self.Actions):
            raise ValueError(f"Action index {action_idx} is outside Env16BC.")
        if self._pending_action is None:
            raise RuntimeError("Call reset() before stepping Env16BC.")
        if action_idx != self._pending_action:
            raise ValueError(
                f"Expected captured action {self._pending_action}, got {action_idx}."
            )

        recorded_key = self._pending_key
        message, next_action, next_key, parsed_next_state = (
            self._receive_valid_action_tick(
                reset=False,
                accept_episode_boundary=True,
            )
        )
        observation, reward, terminated, truncated, info = (
            self._advance_from_world_state(
                message.get("worldState", {}),
                action_idx,
                parsed_next_state=parsed_next_state,
            )
        )
        info = self._minari_info(info)

        self.last_recorded_key = recorded_key
        self.last_recorded_action = action_idx
        if terminated or truncated:
            self._pending_action = None
            self._pending_key = ""
        else:
            if next_action is None:
                raise RuntimeError(
                    "A keyless tick was accepted without ending the episode."
                )
            self._pending_action = next_action
            self._pending_key = next_key

        return observation, reward, terminated, truncated, info
