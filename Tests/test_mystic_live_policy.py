"""Real checkpoint loading and actor replay; no game/server access."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Training.Mystic_Sim.core import Agent, PPOConfig, metadata
from Training.Mystic_Sim.recurrent import RecurrentAgent
from Inference.Mystic_Live.policy import FrozenPolicy
from Inference.Mystic_Live import server


class FrozenPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def checkpoint(self, folder, recurrent=False, encoder='flat', head='linear'):
        env = MysticSimEnv()
        try:
            meta = metadata(env, recurrent=recurrent, actor_head=head, recurrent_encoder=encoder)
        finally:
            env.close()
        model = (RecurrentAgent(meta['observation_scale'], head, encoder) if recurrent
                 else Agent(meta['observation_scale'])).eval()
        path = Path(folder) / 'policy.pt'
        torch.save(dict(format='mystic_sim_ppo_v2', metadata=meta,
                        ppo_config=asdict(PPOConfig(recurrent=recurrent, actor_head=head,
                                                   recurrent_encoder=encoder)),
                        agent=model.state_dict(), optimizer={}, update=0, global_step=0,
                        rng={}, episodes=0, wins=0), path)
        return path, model

    def test_actor_replay_all_supported_architectures(self):
        for recurrent, encoder, head in ((False, 'flat', 'linear'),
                                        (True, 'flat', 'linear'), (True, 'flat', 'tanh'),
                                        (True, 'cross_attention', 'linear'),
                                        (True, 'cross_attention', 'tanh')):
            with self.subTest(recurrent=recurrent, encoder=encoder, head=head), tempfile.TemporaryDirectory() as folder:
                path, reference = self.checkpoint(folder, recurrent, encoder, head)
                with patch.object(MysticSimEnv, 'reset', side_effect=AssertionError('world reset')), \
                     patch.object(torch.optim, 'Adam', side_effect=AssertionError('optimizer created')):
                    live = FrozenPolicy(path)
                self.assertFalse(live.agent.training)
                self.assertTrue(all(not p.requires_grad for p in live.agent.parameters()))
                before = {k: v.clone() for k, v in live.agent.state_dict().items()}
                memory = live.initial_state()
                ref_memory = reference.initial_state(1) if recurrent else None
                rng = np.random.default_rng(19)
                with patch.object(live.agent.critic, 'forward', side_effect=AssertionError('critic called')):
                    for tick in range(4):
                        raw = (rng.random(live.input_size) * live.input_adapter.high).astype(np.float32)
                        tensor = torch.tensor(raw).unsqueeze(0)
                        with torch.inference_mode():
                            if recurrent:
                                logits, _, ref_memory, _ = reference.sequence(
                                    tensor.unsqueeze(0), ref_memory,
                                    torch.tensor([[tick == 0]]))
                                logits = logits[0, 0]
                            else:
                                logits = reference.actor(reference.normalized(tensor))[0]
                        observed_inputs, observed_logits = [], []
                        pre = live.agent.actor.register_forward_pre_hook(
                            lambda module, args: observed_inputs.append(args[0].clone()))
                        post = live.agent.actor.register_forward_hook(
                            lambda module, args, output: observed_logits.append(
                                (output[0] if recurrent else output).clone()))
                        try:
                            choice = live.choose(raw, {}, memory)
                        finally:
                            pre.remove()
                            post.remove()
                        torch.testing.assert_close(observed_inputs[0].reshape(1, -1),
                                                   tensor / reference.observation_scale)
                        torch.testing.assert_close(observed_logits[0].reshape(-1), logits)
                        self.assertEqual(choice.action, int(logits.argmax()))
                        torch.testing.assert_close(torch.tensor(choice.probabilities), logits.softmax(-1))
                        memory = choice.next_memory
                        if recurrent:
                            for actual, expected in zip(memory, ref_memory[:2]):
                                torch.testing.assert_close(actual, expected)
                    for key, value in live.agent.state_dict().items():
                        self.assertTrue(torch.equal(value, before[key]))

    def test_bad_buffer_and_contract_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.checkpoint(folder)
            original = torch.load(path, weights_only=True)
            payload = torch.load(path, weights_only=True)
            payload['agent']['observation_scale'][0] += 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'normalization buffer'):
                FrozenPolicy(path)
            original['metadata']['contract']['actions'][0] = 'attack'
            torch.save(original, path)
            with self.assertRaises(ValueError):
                FrozenPolicy(path)

    def test_sampled_choice_uses_distribution(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.checkpoint(folder)
            live = FrozenPolicy(path, sampled=True)
            with patch('torch.distributions.Categorical.sample', return_value=torch.tensor(6)) as sample:
                choice = live.choose(np.zeros(56, dtype=np.float32), {}, None)
            sample.assert_called_once()
            self.assertEqual(choice.action, 6)

    def test_loaded_actor_lifecycle_does_not_advance_on_duplicate_or_death(self):
        from Tests.test_mystic_live_lifecycle import Clock, request, acknowledge
        from Inference.Mystic_Live.lifecycle import SingleRun
        from Custom_enviornments.Mystic_Sim.actions import ACTIONS
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.checkpoint(folder, recurrent=True)
            actor = FrozenPolicy(path)
            clock = Clock()
            run = SingleRun(actor, input_adapter=actor.input_adapter, monotonic=clock)
            def frame(i):
                result = request(i, clock=clock)
                result['observationContext'] = dict(schema='mystic_live_observation_v1',
                                                    clockId='synthetic-server-clock', stateTimeMs=i*200)
                return result
            with patch.object(actor, 'choose', wraps=actor.choose) as choose:
                first = frame(0)
                reply = run.handle(first)
                self.assertIsNotNone(reply['move'])
                memory = run.memory
                run.handle(first)
                self.assertIs(run.memory, memory)
                clock.advance()
                second = acknowledge(frame(1), reply, label=ACTIONS[run.pending.action])
                run.handle(second)
                self.assertEqual(choose.call_count, 2)
                clock.advance()
                dead = frame(2)
                dead['worldState']['player']['hp'] = 0
                self.assertIsNone(run.handle(dead)['move'])
                self.assertEqual(run.stop_reason, 'player_death')
                self.assertEqual(choose.call_count, 2)

    def test_cli_rejects_unsupported_launch_before_binding(self):
        with patch.object(server.zmq, 'Context') as context:
            for argv in (['--checkpoint', 'missing.pt'],
                         ['--checkpoint', 'missing.pt', '--observe-only'],
                         ['--observe-only', '--sampled']):
                with self.assertRaises(SystemExit) as error:
                    server.main(argv)
                self.assertEqual(error.exception.code, 2)
            context.assert_not_called()

    def test_cli_loads_checkpoint_and_preserves_observation_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.checkpoint(folder)
            with patch.object(server.zmq, 'Context') as context, \
                 patch.object(server, 'serve', side_effect=lambda run, socket, **kw: run.stop()) as serve:
                for name, args in (('actor', ['--checkpoint', str(path)]),
                                   ('observe', ['--observe-only'])):
                    output = Path(folder) / (name + '.json')
                    self.assertEqual(server.main([*args, '--snapshot-output', str(output)]), 0)
                    run = serve.call_args.args[0]
                    self.assertEqual(run.actor is None, name == 'observe')
                    report = json.loads(output.read_text())
                    self.assertEqual('policy' in report, name == 'actor')
                    self.assertEqual(run.progression, 'snapshot')
                context.reset_mock()
                with self.assertRaises(SystemExit):
                    server.main(['--checkpoint', str(path), '--confirmed-feedback', '--input-size', '80'])
                context.assert_not_called()

    def test_cli80_requires_explicit_feedback_before_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.checkpoint(folder, recurrent=True)
            with patch.object(server.zmq, 'Context') as context:
                with self.assertRaises(SystemExit):
                    server.main(['--checkpoint', str(path)])
                context.assert_not_called()
            with patch.object(server.zmq, 'Context'), \
                 patch.object(server, 'serve', side_effect=lambda run, socket, **kw: run.stop()) as serve:
                server.main(['--checkpoint', str(path), '--confirmed-feedback',
                             '--snapshot-output', str(Path(folder) / 'strict.json')])
                self.assertEqual(serve.call_args.args[0].progression, 'confirmed_feedback')


if __name__ == '__main__':
    unittest.main()
