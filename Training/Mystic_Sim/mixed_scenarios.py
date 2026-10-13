"""Episode-level scenario mixture, with fixed probabilities and no advancement gates."""
from copy import deepcopy
from dataclasses import dataclass
import math
import re

import gymnasium as gym
import numpy as np

from Custom_enviornments.Mystic_Sim.config import EncounterConfig
from Custom_enviornments.Mystic_Sim.encounters import scenario_for_encounter


@dataclass(frozen=True)
class ScenarioChoice:
    name: str
    weight: float
    encounter: EncounterConfig | None


@dataclass(frozen=True)
class MixedEncounter(EncounterConfig):
    mixture: tuple[ScenarioChoice, ...] = ()

    def __post_init__(self):
        EncounterConfig.__post_init__(self)
        if not self.mixture or len({c.name for c in self.mixture}) != len(self.mixture):
            raise ValueError('Mixture must contain unique named scenarios')
        for choice in self.mixture:
            if not isinstance(choice.name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', choice.name):
                raise ValueError('Mixture names must use letters, digits, underscores or hyphens')
            if (type(choice.weight) not in (int, float) or not math.isfinite(choice.weight)
                    or choice.weight <= 0):
                raise ValueError('Mixture weights must be finite and positive')
            if isinstance(choice.encounter, MixedEncounter):
                raise ValueError('Nested mixtures are not supported')
        if not math.isfinite(sum(c.weight for c in self.mixture)):
            raise ValueError('Mixture weight sum must be finite')


def mixture_from_dict(values):
    from .curriculum import encounter_from_dict
    values = dict(values)
    choices = values.get('mixture')
    if not isinstance(choices, (list, tuple)):
        raise ValueError('mixture must be a list of named weighted encounters')
    parsed = []
    for choice in choices:
        if not isinstance(choice, dict) or set(choice) != {'name', 'weight', 'encounter'}:
            raise ValueError('Each mixture item needs name, weight and encounter')
        parsed.append(ScenarioChoice(choice['name'], choice['weight'],
                                     encounter_from_dict(choice['encounter'])))
    values['mixture'] = tuple(parsed)
    return MixedEncounter(**values)


class MixedScenarioEnv(gym.Env):
    """Delegate episodes to existing envs; never change a running world's scenario.

    Scenario choice uses its own RNG, independent of combat draws and episode
    duration. The choice is diagnostic metadata, never a policy input.
    """
    def __init__(self, encounter):
        from .curriculum import encounter_factory
        self.mixture = encounter
        self.config = scenario_for_encounter(encounter)
        self.envs = []
        try:
            for choice in encounter.mixture:
                self.envs.append(encounter_factory(choice.encounter)())
        except Exception:
            self.close()
            raise
        self.active = self.envs[0]
        self.choice = encounter.mixture[0]
        self.observation_space = self.active.observation_space
        self.action_space = self.active.action_space
        weights = np.asarray([c.weight for c in encounter.mixture], dtype=float)
        self.probabilities = weights / weights.sum()
        self.sampling_rng = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, 'active'), name)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.sampling_rng is None:
            self.sampling_rng = np.random.default_rng(seed)
        index = int(self.sampling_rng.choice(len(self.envs), p=self.probabilities))
        world_seed = int(self.sampling_rng.integers(0, 2**32))
        self.active, self.choice = self.envs[index], self.mixture.mixture[index]
        obs, info = self.active.reset(seed=world_seed, options=options)
        info['training_scenario'] = self.choice.name
        info['scenario_seed'] = world_seed
        info['scenario_kill_goal'] = self.active.reward_config.win_kills
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.active.step(action)
        info['training_scenario'] = self.choice.name
        info['scenario_kill_goal'] = self.active.reward_config.win_kills
        return obs, reward, terminated, truncated, info

    def sampling_state(self):
        return deepcopy(self.sampling_rng.bit_generator.state)

    def restore_sampling_state(self, state):
        self.sampling_rng = np.random.default_rng()
        self.sampling_rng.bit_generator.state = deepcopy(state)

    def close(self):
        for env in self.envs:
            env.close()
