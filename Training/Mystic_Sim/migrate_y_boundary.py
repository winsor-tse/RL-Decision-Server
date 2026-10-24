"""Explicitly transfer a checkpoint to task-boundary termination semantics."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from Utils.ppo_checkpoint import atomic_torch_save
from .core import load_checkpoint, metadata, saved_config
from .curriculum import checkpoint_factory


# Only this audited engine transition is allowed. Do not bypass arbitrary hash
# mismatches: observations, rewards, map and all other metadata must match.
OLD_ENGINE = 'b4e86a1de9128e95418f6e98d4e571ec006684210e1af87e3d1d32ab4ec703ee'
NEW_ENGINE = '2aa075a347ac5aafa12f6813155f93f7aa7bed1e99c5787952cac32a6182a7db'


def migrate(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination:
        raise ValueError('Choose a new output path; existing checkpoints are never overwritten')
    raw = torch.load(source, map_location='cpu', weights_only=True)
    config = saved_config(raw)
    env = checkpoint_factory(raw)()
    try:
        target = metadata(env, recurrent=config.recurrent, actor_head=config.actor_head, recurrent_encoder=config.recurrent_encoder, history_mode=config.history_mode)
    finally:
        env.close()
    if target['engine_sha256'] != NEW_ENGINE:
        raise ValueError('Current engine is not the audited Y-boundary termination version')
    expected_source = deepcopy(target)
    expected_source['engine_sha256'] = OLD_ENGINE
    payload = load_checkpoint(source, expected_source)
    payload['metadata'] = target
    payload['y_boundary_migration'] = {
        'source_checkpoint': str(source), 'old_engine_sha256': OLD_ENGINE,
        'new_engine_sha256': NEW_ENGINE,
        'change': 'y_boundary: truncated -> terminated loss; existing positioning penalty retained; no extra death penalty',
        'weights_optimizer_rng_preserved': True,
        'historical_metrics': 'episodes/wins belong to the old task; evaluate on the new task before comparing',
    }
    atomic_torch_save(payload, destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    result = migrate(root / args.checkpoint, root / args.output)
    print(f'Migrated checkpoint: {result}\nWeights/optimizer/RNG retained; no training performed.')


if __name__ == '__main__':
    main()
