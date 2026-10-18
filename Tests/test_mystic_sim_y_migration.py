from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from Training.Mystic_Sim.core import make_env, metadata, PPOConfig, load_checkpoint
from Training.Mystic_Sim.migrate_y_boundary import migrate, OLD_ENGINE, NEW_ENGINE


class MigrationTests(unittest.TestCase):
    def test_preserves_payload_and_rejects_other_changes(self):
        env = make_env()
        try:
            current = metadata(env)
        finally:
            env.close()
        # Exercise the historical migration against its audited target, not the
        # expanded observation engine. Production migration stays hash-locked.
        actual = deepcopy(current)
        current['engine_sha256'] = NEW_ENGINE
        metadata_patch = patch('Training.Mystic_Sim.migrate_y_boundary.metadata', return_value=current)
        old = deepcopy(current)
        old['engine_sha256'] = OLD_ENGINE
        payload = dict(format='mystic_sim_ppo_v2', metadata=old,
                       ppo_config=asdict(PPOConfig()), agent={'weight': torch.tensor([3.])},
                       optimizer={'sentinel': 4}, rng={'sentinel': 5},
                       update=1, global_step=1024, episodes=2, wins=1)
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'old.pt', Path(folder)/'new.pt'
            torch.save(payload, source)
            original = source.read_bytes()
            with self.assertRaises(ValueError):
                load_checkpoint(source, current)
            self.assertNotEqual(actual['engine_sha256'], NEW_ENGINE)
            with self.assertRaisesRegex(ValueError, 'not the audited'):
                migrate(source, target)
            self.assertFalse(target.exists())
            with metadata_patch:
                migrate(source, target)
            new = load_checkpoint(target, current)
            self.assertTrue(torch.equal(new['agent']['weight'], payload['agent']['weight']))
            for key in ('optimizer', 'rng', 'update', 'global_step', 'episodes', 'wins'):
                self.assertEqual(new[key], payload[key])
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(ValueError):
                migrate(source, target)
            payload['metadata']['map_sha256'] = 'wrong'
            torch.save(payload, source)
            with patch('Training.Mystic_Sim.migrate_y_boundary.metadata', return_value=current), self.assertRaises(ValueError):
                migrate(source, Path(folder)/'invalid.pt')


if __name__ == '__main__':
    unittest.main()
