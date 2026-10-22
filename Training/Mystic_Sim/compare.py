"""Evaluate frozen checkpoints across configured stages; never train."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .core import json_write, metadata
from .curriculum import encounter_factory, validate_transfer_metadata
from .curriculum_run import load_run_config
from .evaluate import evaluate, evaluation_line, load_evaluation_agent


def load_comparison(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    allowed = {'version', 'stage_config', 'stages', 'policies', 'episodes', 'seed',
               'modes', 'device', 'output'}
    if not isinstance(data, dict) or set(data) - allowed or data.get('version') != 1:
        raise ValueError('Expected comparison config version 1 with supported fields')
    for key in ('stage_config', 'output'):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f'{key} must be a nonempty path')
    for key in ('stages', 'modes'):
        values = data.get(key, ['sampled', 'greedy'] if key == 'modes' else None)
        if (not isinstance(values, list) or not values
                or any(not isinstance(x, str) or not x.strip() for x in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'{key} must contain unique nonempty names')
        data[key] = values
    if set(data['modes']) - {'sampled', 'greedy'}:
        raise ValueError('modes must be sampled and/or greedy')
    policies = data.get('policies')
    if (not isinstance(policies, dict) or not policies
            or any(not name.strip() or not isinstance(value, str) or not value.strip()
                   for name, value in policies.items())):
        raise ValueError('policies must map labels to checkpoint paths')
    for key, default, minimum in (('episodes', 30, 1), ('seed', 22000, 0)):
        value = data.setdefault(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f'Invalid {key}')
    data.setdefault('device', 'cpu')
    if data['device'] not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    # Stage files follow config-relative inheritance; checkpoint/output paths
    # follow the existing training convention (relative to the working directory).
    runs = {stage: load_run_config(path.parent / data['stage_config'], stage)
            for stage in data['stages']}
    return data, runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true',
                        help='Check checkpoints, contracts and resets without episode rollouts')
    args = parser.parse_args(argv)
    data, runs = load_comparison(args.config)
    episodes = args.episodes if args.episodes is not None else data['episodes']
    seed = args.seed if args.seed is not None else data['seed']
    if episodes < 1 or seed < 0:
        parser.error('episodes must be positive and seed nonnegative')
    if data['device'] == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(1)
    seeds = list(range(seed, seed + episodes))
    prepared = []
    # Validate every pairing before running any episodes or writing results.
    for label, checkpoint in data['policies'].items():
        agent, _, source = load_evaluation_agent(checkpoint, data['device'])
        factories = {}
        for stage, run in runs.items():
            factory = encounter_factory(run.encounter)
            env = factory()
            try:
                target = metadata(env, recurrent=bool(getattr(agent, 'recurrent', False)),
                                  actor_head=getattr(agent, 'actor_head', 'linear'),
                                  recurrent_encoder=getattr(agent, 'recurrent_encoder', 'flat'))
                validate_transfer_metadata(source, target)
                env.reset(seed=seed)
            finally:
                env.close()
            factories[stage] = factory
        prepared.append((label, checkpoint, agent, factories))
    if args.validate_only:
        print(f'Validated {len(prepared)} policies across {len(runs)} stages; no rollouts or training.')
        return None
    output = args.output or Path(data['output'])
    result = {'training_performed': False, 'seeds': seeds, 'modes': data['modes'],
              'stages': {name: (None if run.encounter is None else
                         asdict(run.encounter))
                         for name, run in runs.items()}, 'results': {}}
    for label, checkpoint, agent, factories in prepared:
        entry = {'checkpoint': str(Path(checkpoint).resolve()), 'stages': {}}
        result['results'][label] = entry
        for stage, factory in factories.items():
            entry['stages'][stage] = {}
            for mode in data['modes']:
                print(f'Evaluating {label} / {stage} / {mode} ({episodes} episodes)...', flush=True)
                report = evaluate(agent, seeds, device=data['device'],
                                  stochastic=mode == 'sampled', factory=factory)
                entry['stages'][stage][mode] = report
                print(evaluation_line(f'{label} / {stage} / {mode}', report), flush=True)
                json_write(output, result)
    print('Comparison:', output.resolve())
    return result


if __name__ == '__main__':
    main()
