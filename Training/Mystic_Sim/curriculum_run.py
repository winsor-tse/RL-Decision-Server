"""Run ONE configured curriculum encounter. Stage changes are always explicit."""
import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import torch

from .core import PPOConfig, saved_config
from .curriculum import (encounter_factory, encounter_from_dict, load_stage_checkpoint,
                         stage_schedule, VALIDATION_SEED_BANKS)
from .core import metadata, load_checkpoint
from .train import train


@dataclass(frozen=True)
class ManualRun:
    stage: str
    encounter: object
    ppo: PPOConfig
    runtime: dict
    schedule_timesteps: int | None
    continue_from: Path | None = None


def read_run_config(path, ancestors=()):
    """Inherit shared PPO/stage settings; extends paths are relative to this file."""
    path = Path(path).resolve()
    if path in ancestors or len(ancestors) >= 16:
        raise ValueError('Cyclic or excessively deep curriculum config inheritance')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('Curriculum config must be an object')
    parent = data.pop('extends', None)
    if parent is None:
        return data
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError('extends must be a nonempty config filename')
    base = read_run_config(path.parent / parent, (*ancestors, path))

    def merge(base, override):
        result = dict(base)
        for key, value in override.items():
            result[key] = (merge(result[key], value)
                           if isinstance(result.get(key), dict) and isinstance(value, dict) else value)
        return result

    return merge(base, data)


def load_run_config(path, stage=None):
    data = read_run_config(path)
    if not isinstance(data, dict) or set(data) - {
            'version', 'stage', 'stages', 'ppo', 'runtime', 'schedule_timesteps', 'continue_from'}:
        raise ValueError('Unknown manual curriculum config fields')
    if type(data.get('version')) is not int or data['version'] != 1:
        raise ValueError('Expected manual curriculum config version 1')
    stages = data.get('stages')
    if not isinstance(stages, dict) or not stages:
        raise ValueError('stages must be a nonempty object')
    encounters = {}
    for name, values in stages.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Stage names must be nonempty strings')
        if values is None:
            if name != 'full':
                raise ValueError('Only full may use null for the original environment')
            encounters[name] = None
        else:
            if not isinstance(values, dict) or 'stage_id' in values:
                raise ValueError('Stage definitions use the stage name, not a nested stage_id')
            encounters[name] = encounter_from_dict(dict(values, stage_id=name))
    selected = stage if stage is not None else data.get('stage')
    if selected not in encounters:
        raise ValueError(f'Unknown stage {selected!r}; choose from {list(encounters)}')
    ppo_values = data.get('ppo', {})
    if not isinstance(ppo_values, dict):
        raise ValueError('ppo must be an object')
    defaults = PPOConfig()
    for field in fields(defaults):
        if field.name not in ppo_values:
            continue
        value, default = ppo_values[field.name], getattr(defaults, field.name)
        if default is None:
            valid = value is None or type(value) in (int, float)
        elif type(default) is float:
            valid = type(value) in (int, float)
        else:
            valid = type(value) is type(default)
        if not valid:
            raise ValueError(f'Invalid JSON type for PPO field {field.name}')
    try:
        ppo = PPOConfig(**ppo_values)
    except TypeError as exc:
        raise ValueError(f'Invalid PPO configuration: {exc}') from exc
    ppo.validate()
    runtime = data.get('runtime', {})
    if not isinstance(runtime, dict) or set(runtime) - {
            'device', 'run_dir', 'record_gameplay', 'record_stride', 'torch_threads',
            'stop_after_checkpoint', 'record_from_checkpoint', 'evaluate_full_scenario'}:
        raise ValueError('Unknown runtime configuration fields')
    runtime = dict(device='auto', record_gameplay=True, record_stride=5, torch_threads=1,
                   record_from_checkpoint=1, evaluate_full_scenario=True) | runtime
    if runtime['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    for name in ('record_stride', 'torch_threads'):
        if type(runtime[name]) is not int or runtime[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(runtime['record_gameplay']) is not bool:
        raise ValueError('record_gameplay must be boolean')
    if type(runtime['evaluate_full_scenario']) is not bool:
        raise ValueError('evaluate_full_scenario must be boolean')
    first = runtime['record_from_checkpoint']
    if type(first) is not int or not 1 <= first <= 10:
        raise ValueError('record_from_checkpoint must be in 1..10')
    stop = runtime.get('stop_after_checkpoint')
    if stop is not None and (type(stop) is not int or not 1 <= stop <= 10):
        raise ValueError('stop_after_checkpoint must be in 1..10')
    if 'run_dir' in runtime:
        if not isinstance(runtime['run_dir'], str) or not runtime['run_dir'].strip():
            raise ValueError('run_dir must be a nonempty string')
        runtime['run_dir'] = Path(runtime['run_dir'].format(stage=selected, seed=ppo.seed))
    schedule = data.get('schedule_timesteps')
    if schedule is not None and (type(schedule) is not int or schedule <= 0):
        raise ValueError('schedule_timesteps must be a positive integer')
    source = data.get('continue_from')
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError('continue_from must be a checkpoint path or null')
    return ManualRun(selected, encounters[selected], ppo, runtime, schedule,
                     Path(source) if source is not None else None)


def preflight(run, *, resume=None, continue_from=None):
    """Validate geometry/contracts/budgets without collecting training or updating weights."""
    if resume and continue_from:
        raise ValueError('Choose resume or continue_from, not both')
    env = encounter_factory(run.encounter)()
    try:
        contract = metadata(env, recurrent=run.ppo.recurrent, actor_head=run.ppo.actor_head)
        payload = load_checkpoint(resume, contract) if resume else None
        if payload and asdict(saved_config(payload)) != asdict(run.ppo):
            raise ValueError('Resume config must match saved PPO settings; use continuation for a new budget')
        source = load_stage_checkpoint(continue_from, contract, run.ppo) if continue_from else None
        schedule = stage_schedule(run.ppo, resume_payload=payload, source_payload=source,
                                  schedule_timesteps=run.schedule_timesteps)
        if continue_from:
            schedule['source_checkpoint'] = str(Path(continue_from).resolve())
        completed = payload['update'] if payload else 0
        if completed >= run.ppo.updates:
            raise ValueError('Resume checkpoint already completed; use --continue-from for a new run')
        initial_seed = run.ppo.seed + completed * run.ppo.num_envs
        training_seeds = range(initial_seed, initial_seed + run.ppo.num_envs)
        evaluation_seeds = range(run.ppo.eval_seed, run.ppo.eval_seed + run.ppo.eval_episodes)
        reserved = set(evaluation_seeds).union(*VALIDATION_SEED_BANKS)
        if set(training_seeds).intersection(reserved):
            raise ValueError('Training and evaluation seed ranges overlap')
        # Only reset: there are no rollout steps, agent construction or optimizer calls.
        for seed in training_seeds:
            env.reset(seed=seed)
        # A random reset sample need not select every mixture member. Validate
        # each member's geometry explicitly without collecting any transitions.
        from .mixed_scenarios import MixedEncounter
        if isinstance(run.encounter, MixedEncounter):
            schedule['mode'] = 'fixed_episode_mixture'
            total_weight = sum(c.weight for c in run.encounter.mixture)
            schedule['episode_probabilities'] = {c.name: c.weight/total_weight for c in run.encounter.mixture}
            for choice in run.encounter.mixture:
                member = encounter_factory(choice.encounter)()
                try:
                    member.reset(seed=initial_seed)
                finally:
                    member.close()
        return {'stage': run.stage, 'encounter': contract['scenario']['encounter'],
                'local_timesteps': run.ppo.updates * run.ppo.batch_size,
                'schedule': schedule, 'evaluation_seeds': list(evaluation_seeds),
                'full_scenario_evaluation': run.runtime['evaluate_full_scenario'],
                'record_from_checkpoint': run.runtime['record_from_checkpoint'],
                'automatic_advancement': False}
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', help='Override the selected stage in the JSON config')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', type=Path, help='Resume the same stage/budget and saved optimizer')
    group.add_argument('--continue-from', type=Path, help='New run, retaining learned weights and optimizer')
    group.add_argument('--fresh', action='store_true', help='Ignore the configured checkpoint and initialize fresh weights')
    parser.add_argument('--list-stages', action='store_true', help='List configured encounters without training or loading checkpoints')
    parser.add_argument('--validate-only', action='store_true', help='Validate/reset only; never train or write run files')
    args = parser.parse_args(argv)
    run = load_run_config(args.config, args.stage)
    if args.list_stages:
        rows = []
        for name in read_run_config(args.config)['stages']:
            stage = load_run_config(args.config, name)
            rows.append({'stage': name, 'selected': name == run.stage,
                         'enemy_count': stage.encounter.enemy_count if stage.encounter else 80,
                         'kill_goal': stage.encounter.kill_goal if stage.encounter else 5,
                         'path_distance': stage.encounter.path_distance if stage.encounter else None})
        print(json.dumps(rows, indent=2))
        return rows
    source = (None if args.resume or args.fresh else
              args.continue_from if args.continue_from is not None else run.continue_from)
    report = preflight(run, resume=args.resume, continue_from=source)
    report['initialization'] = 'resume' if args.resume else 'continue' if source else 'fresh'
    print(json.dumps(report, indent=2))
    if args.validate_only:
        return report
    runtime = dict(run.runtime)
    torch.set_num_threads(runtime.pop('torch_threads'))
    if args.run_dir is not None:
        runtime['run_dir'] = args.run_dir
    elif args.resume:
        runtime.pop('run_dir', None)  # Resume defaults to the checkpoint's own run.
    if args.device is not None:
        runtime['device'] = args.device
    return train(run.ppo, encounter=run.encounter, resume=args.resume,
                 continue_from=source, schedule_timesteps=run.schedule_timesteps,
                 **runtime)


if __name__ == '__main__':
    main()
