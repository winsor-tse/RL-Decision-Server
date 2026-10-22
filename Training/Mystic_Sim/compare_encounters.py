"""Evaluate a fixed checkpoint and random baseline on paired encounter seeds; never train."""
import argparse
from pathlib import Path
import torch

from .core import Agent, metadata, load_checkpoint, saved_config, json_write
from .curriculum import checkpoint_factory, encounter_factory, validate_transfer_metadata
from .curriculum_run import load_run_config
from .evaluate import evaluate
from .recurrent import RecurrentAgent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', default=['three_innies', 'three_innies_tight'])
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0:
        parser.error('Require positive episodes and nonnegative seed')
    if args.output.exists():
        parser.error('Choose a new output path; existing comparisons are preserved')
    torch.set_num_threads(1)
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = saved_config(raw)
    env = checkpoint_factory(raw)()
    try:
        contract = metadata(env, recurrent=config.recurrent, actor_head=config.actor_head, recurrent_encoder=config.recurrent_encoder)
        payload = load_checkpoint(args.checkpoint, contract)
    finally:
        env.close()
    agent = (RecurrentAgent(contract['observation_scale'], config.actor_head, config.recurrent_encoder)
             if config.recurrent else Agent(contract['observation_scale']))
    agent.load_state_dict(payload['agent']); agent.eval()
    report = {'checkpoint': str(args.checkpoint.resolve()), 'training_performed': False, 'stages': {}}
    for stage in args.stages:
        run = load_run_config(args.config, stage)
        factory = encounter_factory(run.encounter)
        env = factory()
        try:
            target = metadata(env, recurrent=config.recurrent, actor_head=config.actor_head, recurrent_encoder=config.recurrent_encoder)
            validate_transfer_metadata(contract, target)
        finally:
            env.close()
        results = {}
        for mode in ('random', 'greedy', 'sampled'):
            results[mode] = evaluate(None if mode == 'random' else agent,
                                     list(range(args.seed, args.seed + args.episodes)),
                                     stochastic=mode == 'sampled', factory=factory)
            print(stage, mode, results[mode]['summary'], flush=True)
        report['stages'][stage] = {'scenario': target['scenario'], 'results': results}
    json_write(args.output, report)
    return report


if __name__ == '__main__':
    main()
