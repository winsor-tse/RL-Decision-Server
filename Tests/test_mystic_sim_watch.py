"""Frozen policy viewer must match evaluation, including recurrent history/reset."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import unittest
from pathlib import Path
import tempfile

import torch

from Custom_enviornments.Mystic_Sim.config import ScenarioConfig
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Training.Mystic_Sim.core import metadata
from Training.Mystic_Sim.recurrent import RecurrentAgent
from Training.Mystic_Sim.evaluate import evaluate
from Training.Mystic_Sim.watch import PolicySession, parse_watch_args


class WatchTests(unittest.TestCase):
    def test_checkpoint_cli_forms_and_missing_subfolder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'run with spaces/checkpoints/checkpoint_09.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            relative = checkpoint.relative_to(root).as_posix()
            for argv in ([relative, '--seed', '2000'],
                         ['--seed', '2000', '--checkpoint', relative],
                         ['--seed', '2000', 'checkpoint', relative],
                         ['checkpoint', '--seed', '2000', relative],
                         ['--seed', '2000', '--checkpoint', str(checkpoint)]):
                args = parse_watch_args(argv, root=root)
                self.assertEqual(args.checkpoint, checkpoint.resolve())
                self.assertEqual(args.seed, 2000)
            with redirect_stdout(io.StringIO()) as output:
                args = parse_watch_args(['run with spaces/checkpoint_09.pt'], root=root)
            self.assertEqual(args.checkpoint, checkpoint.resolve())
            self.assertIn('checkpoints subfolder', output.getvalue())
            for argv in ([], ['missing.pt'], [relative, '--checkpoint', relative],
                         [relative, relative], [relative, '--seed', '-1']):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_watch_args(argv, root=root)
                self.assertEqual(error.exception.code, 2)

    def test_evaluation_parity_and_replay(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        config = ScenarioConfig()
        config = replace(config, reward=replace(config.reward, max_episode_steps=30))
        def factory():
            return MysticSimEnv(config=config)
        env = factory()
        try:
            scale = metadata(env, recurrent=True, actor_head='linear')['observation_scale']
        finally:
            env.close()
        torch.manual_seed(5)
        agent = RecurrentAgent(scale, 'linear').eval()
        for mode in ('sampled', 'greedy'):
            with self.subTest(mode=mode):
                expected = evaluate(agent, [20000], factory=factory,
                                    stochastic=mode == 'sampled')['episodes'][0]
                session = PolicySession(agent, factory, seed=20000, mode=mode)
                try:
                    self.assertIsNone(session.step())
                    traces = []
                    for _ in range(2):
                        session.paused = False
                        actions = []
                        with redirect_stdout(io.StringIO()):
                            while not session.env.episode_done:
                                session.step(0)  # Ignored: cannot override policy.
                                actions.append(session.last_action)
                        self.assertAlmostEqual(session.total_reward, expected['episode_return'])
                        self.assertEqual(session.env.world.step_count, expected['length'])
                        self.assertEqual(session.info['episode_end_reason'], expected['end_reason'])
                        traces.append(actions)
                        self.assertIsNone(session.step())
                        session.reset()
                    self.assertEqual(traces[0], traces[1])
                    session.reset(new_seed=True)
                    self.assertEqual(session.seed, 20001)
                    self.assertTrue(session.episode_start)
                    self.assertEqual(session.env.world.step_count, 0)
                finally:
                    session.close()


if __name__ == '__main__':
    unittest.main()
