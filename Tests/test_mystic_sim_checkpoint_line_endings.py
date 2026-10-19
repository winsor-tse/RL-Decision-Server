"""Only byte-verified line-ending differences may bypass the raw engine hash."""
import copy
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

import torch

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Training.Mystic_Sim.core import load_checkpoint, metadata


class CheckpointLineEndingsTests(unittest.TestCase):
    def test_line_endings_accepted_but_code_and_contract_mismatches_rejected(self):
        env = MysticSimEnv()
        self.addCleanup(env.close)
        expected = metadata(env, recurrent=True)
        source = Path(inspect.getfile(MysticSimEnv)).parent
        files = [p for p in sorted(source.glob('*.py')) if p.name != 'viewer.py']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            for newline in (b'\n', b'\r\n'):
                saved = copy.deepcopy(expected)
                saved['engine_sha256'] = hashlib.sha256(b''.join(
                    p.name.encode() + p.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', newline)
                    for p in files)).hexdigest()
                payload = dict(format='mystic_sim_ppo_v2', metadata=saved,
                               agent={}, optimizer={}, ppo_config={}, update=1,
                               global_step=1024, rng={}, episodes=1, wins=0)
                torch.save(payload, path)
                before = path.read_bytes()
                self.assertEqual(load_checkpoint(path, expected)['metadata'], saved)
                self.assertEqual(path.read_bytes(), before)
                for field in ('engine_sha256', 'map_sha256', 'contract'):
                    changed = copy.deepcopy(payload)
                    changed['metadata'][field] = 'different'
                    torch.save(changed, path)
                    with self.assertRaisesRegex(ValueError, 'incompatible'):
                        load_checkpoint(path, expected)
                # A caller requesting another engine must not gain compatibility.
                torch.save(payload, path)
                wrong = dict(expected, engine_sha256='unrelated historical engine')
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    load_checkpoint(path, wrong)
