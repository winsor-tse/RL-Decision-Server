"""Manual curriculum stages and compatible transfer; no automatic progression."""
from copy import deepcopy
from dataclasses import dataclass

from Custom_enviornments.Mystic_Sim.config import EncounterConfig
from Custom_enviornments.Mystic_Sim.encounters import scenario_for_encounter
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from .core import make_env, load_checkpoint, saved_config


@dataclass(frozen=True)
class EncounterFactory:
    """Pickleable factory for existing SimBatch and evaluate(factory=...)."""
    encounter: EncounterConfig

    def __post_init__(self):
        scenario_for_encounter(self.encounter)  # Fail before creating workers.

    def __call__(self):
        from .nearby_areas import NearbyAreasEncounter, NearbyAreasEnv
        env_type = NearbyAreasEnv if isinstance(self.encounter, NearbyAreasEncounter) else MysticSimEnv
        return env_type(config=scenario_for_encounter(self.encounter), trace=False)


# Never derive the evaluation baseline from the current training stage.
make_full_evaluation_env = make_env

# Separate validation banks for the provisional two-evaluation advancement gate.
# No progression/controller is enabled by these constants.
VALIDATION_SEED_BANKS = (tuple(range(20000, 20030)), tuple(range(21000, 21030)))
ADVANCEMENT_SUCCESSES = 24
ADVANCEMENT_EPISODES = 30
ADVANCEMENT_CONSECUTIVE_EVALUATIONS = 2


def encounter_from_dict(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('encounter must be an object or null (original full scenario)')
    values = dict(value)
    if 'path_distance' in values:
        if not isinstance(values['path_distance'], (list, tuple)):
            raise ValueError('path_distance must be a two-value array')
        values['path_distance'] = tuple(values['path_distance'])
    try:
        if 'mixture' in values:
            from .mixed_scenarios import mixture_from_dict
            return mixture_from_dict(values)
        if 'placement' in values or 'spawn_area_count' in values:
            from .nearby_areas import NearbyAreasEncounter
            return NearbyAreasEncounter(**values)
        return EncounterConfig(**values)
    except TypeError as exc:
        raise ValueError(f'Invalid encounter configuration: {exc}') from exc


def encounter_factory(encounter):
    from .mixed_scenarios import MixedEncounter, MixedScenarioEnv
    from functools import partial
    if isinstance(encounter, MixedEncounter):
        return partial(MixedScenarioEnv, encounter)
    return make_env if encounter is None else EncounterFactory(encounter)


def checkpoint_factory(payload):
    return encounter_factory(encounter_from_dict(payload['metadata']['scenario'].get('encounter')))


def validate_transfer_metadata(source, target):
    """Permit only initial encounter and victory-goal differences, never engine drift."""
    def invariant(value):
        value = deepcopy(value)
        value['scenario'].pop('encounter', None)
        value['scenario']['reward'].pop('win_kills')
        return value
    if invariant(source) != invariant(target):
        raise ValueError('Stage continuation may change only encounter settings and kill goal; '
                         'map, engine, rewards, observations and policy must remain compatible')


def load_stage_checkpoint(path, target_metadata, config):
    import torch
    from .core import metadata, checkpoint_updates
    raw = torch.load(path, map_location='cpu', weights_only=True)
    env = checkpoint_factory(raw)()
    old_config = saved_config(raw)
    try:
        source_metadata = metadata(env, recurrent=old_config.recurrent, actor_head=old_config.actor_head, recurrent_encoder=old_config.recurrent_encoder)
    finally:
        env.close()
    payload = load_checkpoint(path, source_metadata)
    validate_transfer_metadata(source_metadata, target_metadata)
    # Explicit continuations may change the learning rate while retaining Adam
    # moments. train() applies the configured rate before the first PPO update.
    # Resume remains strict; all other PPO update settings stay compatible.
    from dataclasses import asdict
    old, new = asdict(old_config), asdict(config)
    for key in ('total_timesteps', 'seed', 'eval_seed', 'eval_episodes', 'learning_rate'):
        old.pop(key); new.pop(key)
    if old != new:
        raise ValueError('Stage continuation requires matching PPO settings except learning rate, budget and seeds/evaluation count')
    if (type(payload['update']) is not int or type(payload['global_step']) is not int
            or payload['update'] not in checkpoint_updates(old_config.updates)
            or payload['global_step'] != payload['update'] * old_config.batch_size):
        raise ValueError('Invalid source checkpoint progress')
    return payload


def stage_schedule(config, *, resume_payload=None, source_payload=None, schedule_timesteps=None):
    """Separate local run budget from the shared learning-rate horizon and progress."""
    if resume_payload is not None and source_payload is not None:
        raise ValueError('Choose resume or stage continuation, not both')
    previous = resume_payload if resume_payload is not None else source_payload
    saved = previous.get('curriculum', {}) if previous else {}
    offset = saved.get('step_offset', 0)
    if source_payload is not None:
        offset += source_payload['global_step']
    horizon = saved.get('schedule_timesteps')
    if previous and horizon is None:
        old = saved_config(previous)
        horizon = old.updates * old.batch_size
    if horizon is not None and schedule_timesteps is not None and horizon != schedule_timesteps:
        raise ValueError('Keep the saved learning-rate horizon across resume/stage continuation')
    horizon = horizon if horizon is not None else (
        schedule_timesteps if schedule_timesteps is not None else config.updates * config.batch_size)
    if type(horizon) is not int or horizon <= 0:
        raise ValueError('schedule_timesteps must be a positive integer')
    if config.anneal_lr and offset + config.updates * config.batch_size > horizon:
        raise ValueError('Run exceeds the shared learning-rate horizon; plan a larger horizon '
                         'at the first stage or start the curriculum with anneal_lr=false')
    return {'step_offset': offset, 'schedule_timesteps': horizon,
            'source_checkpoint': saved.get('source_checkpoint') if resume_payload else None,
            'mode': 'manual', 'automatic_advancement': False}


def learning_rate(config, update, schedule):
    before = schedule['step_offset'] + (update - 1) * config.batch_size
    return (config.learning_rate * (1 - before / schedule['schedule_timesteps'])
            if config.anneal_lr else config.learning_rate)
