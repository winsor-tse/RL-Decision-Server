"""Phase 0/1 contract and factories; progression is deliberately a later phase."""
from dataclasses import dataclass

from Custom_enviornments.Mystic_Sim.config import EncounterConfig
from Custom_enviornments.Mystic_Sim.encounters import scenario_for_encounter
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from .core import make_env


@dataclass(frozen=True)
class EncounterFactory:
    """Pickleable factory for existing SimBatch and evaluate(factory=...)."""
    encounter: EncounterConfig

    def __post_init__(self):
        scenario_for_encounter(self.encounter)  # Fail before creating workers.

    def __call__(self):
        return MysticSimEnv(config=scenario_for_encounter(self.encounter), trace=False)


# Never derive the evaluation baseline from the current training stage.
make_full_evaluation_env = make_env

# Separate validation banks for the provisional two-evaluation advancement gate.
# No progression/controller is enabled by these constants.
VALIDATION_SEED_BANKS = (tuple(range(20000, 20030)), tuple(range(21000, 21030)))
ADVANCEMENT_SUCCESSES = 24
ADVANCEMENT_EPISODES = 30
ADVANCEMENT_CONSECUTIVE_EVALUATIONS = 2
