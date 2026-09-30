"""Versioned action contracts; legacy orderings must never be inferred by size."""
from enum import IntEnum
from numbers import Integral


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ATTACK = 4
    ARCANE_BLAST = 5
    ACID_CLOUD = 6
    TEMPEST_INFERNO = 7


ACTIONS = ("up", "down", "left", "right", "attack",
           "castSpell:1", "castSpell:2", "castSpell:3")
ACTION_SCHEMA = "mystic-eight-v1"
OBSERVATION_SCHEMA = "mystic-26-v1"
DATASET_ID = "mystic/BC-v1"
LEGACY_LIVE_ACTIONS = ACTIONS + ("castSpell:5", "castSpell:6", "castSpell:7")
LEGACY_BC_ACTIONS = ("up", "left", "right", "down") + LEGACY_LIVE_ACTIONS[4:]


def action_from_label(label):
    try:
        return Action(ACTIONS.index(label))
    except ValueError as exc:
        raise ValueError(f"Unsupported Mystic action label: {label!r}") from exc


def action_label(index):
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise ValueError(f"Action must be an integer, got {index!r}")
    return ACTIONS[Action(int(index))]


def remap_legacy_bc(actions):
    """Strict mapping: reject removed spells instead of dropping transitions."""
    result = []
    for position, index in enumerate(actions):
        if (isinstance(index, bool) or not isinstance(index, Integral)
                or not 0 <= index < len(LEGACY_BC_ACTIONS)):
            raise ValueError(f"actions[{position}]: invalid legacy BC index {index!r}")
        label = LEGACY_BC_ACTIONS[int(index)]
        if label not in ACTIONS:
            raise ValueError(f"actions[{position}]: removed action {label}; cannot migrate episode")
        result.append(int(action_from_label(label)))
    return result


def contract_metadata():
    """JSON-safe metadata to embed in new checkpoint/dataset envelopes."""
    return {"action_schema": ACTION_SCHEMA, "actions": list(ACTIONS),
            "observation_schema": OBSERVATION_SCHEMA, "observation_size": 26}


def validate_contract(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Missing Mystic contract metadata; legacy artifacts require migration")
    for key, expected in contract_metadata().items():
        if metadata.get(key) != expected:
            raise ValueError(f"Incompatible {key}: expected {expected!r}, got {metadata.get(key)!r}")
