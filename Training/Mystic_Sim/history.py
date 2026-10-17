"""Causal history and bounded static-terrain sensing; no hidden cooldowns.

Live ZMQ must supply confirmed action outcomes and elapsed time before this
contract can be used there. A sent command is not a confirmed successful cast.
"""
import numpy as np

HISTORY_SCHEMA = 'mystic-local-terrain-50-v3'
SENSOR_RANGE = 4
SENSOR_DIRECTIONS = (('up', 0, -1), ('down', 0, 1), ('left', -1, 0), ('right', 1, 0))
HISTORY_FEATURES = ([f'previous_action_{i}' for i in range(7)] +
                    ['action_applied', 'movement_blocked', 'cooldown_rejected'] +
                    [f'cast_age_{i}_capped_10s' for i in range(1, 4)] +
                    [f'cast_seen_{i}' for i in range(1, 4)] +
                    [f'terrain_{name}_{field}' for name, _, _ in SENSOR_DIRECTIONS
                     for field in ('free_distance', 'detected')])


def terrain_sensors(x, y, *, width, height, blocked_cells, radius=SENSOR_RANGE):
    """Four world-axis rays, ignoring entities; boundaries count as solid terrain.

    Inspect offsets 1..radius, stop at the first blocker. Distance is the number
    of free tiles BEFORE that blocker / radius. No detection gives (1, 0).
    Adjacent blocker gives (0, 1); blocker at offset 4 gives (0.75, 1).
    Only these eight local values enter the policy, not the full collision map.
    """
    if type(radius) is not int or radius < 1:
        raise ValueError('sensor radius must be a positive integer')
    result = np.empty(8, dtype=np.float32)
    for i, (_, dx, dy) in enumerate(SENSOR_DIRECTIONS):
        result[2*i:2*i+2] = (1, 0)
        for distance in range(1, radius + 1):
            tx, ty = x + dx * distance, y + dy * distance
            if not (0 <= tx < width and 0 <= ty < height) or (tx, ty) in blocked_cells:
                result[2*i:2*i+2] = ((distance - 1) / radius, 1)
                break
    return result


def observation_high(base):
    return np.concatenate((base, np.ones(len(HISTORY_FEATURES), dtype=np.float32))).astype(np.float32)


class ObservableHistory:
    def __init__(self, base_high, *, blocked_cells):
        self.width, self.height = int(base_high[0]) + 1, int(base_high[1]) + 1
        self.blocked_cells = frozenset(blocked_cells)
        self.reset()

    @classmethod
    def from_env(cls, env):
        # Use the same static collision layer as movement, not NPC occupancy.
        return cls(env.observation_space.high, blocked_cells=(
            env.map_definition.blocked_cells if env.config.terrain_collision else ()))

    def reset(self):
        self.features = np.zeros(16, dtype=np.float32)

    def encode(self, observation):
        sensors = terrain_sensors(int(observation[0]), int(observation[1]),
                                  width=self.width, height=self.height, blocked_cells=self.blocked_cells)
        return np.concatenate((observation, self.features, sensors)).astype(np.float32)

    def update(self, previous_observation, action, *, applied, blocked,
               cooldown_rejected, elapsed_ms=200):
        # The action occurs at the beginning of this decision interval. Thus a
        # confirmed cast is already elapsed_ms old in the resulting observation.
        self.features[10:13] = np.minimum(1., self.features[10:13] + elapsed_ms / 10000)
        self.features[:10] = 0
        self.features[action] = 1
        self.features[7:10] = (applied, blocked, cooldown_rejected)
        if action >= 4 and applied:
            self.features[10 + action - 4] = min(1., elapsed_ms / 10000)
            self.features[13 + action - 4] = 1

    def update_from_info(self, observation, action, info, elapsed_ms=200):
        self.update(observation, action, applied=bool(info['action_applied']),
                    blocked=info['collision_kind'] is not None,
                    cooldown_rejected=info['action_failure_reason'] in
                    ('cooldown', 'family_cooldown', 'global_cooldown'), elapsed_ms=elapsed_ms)
