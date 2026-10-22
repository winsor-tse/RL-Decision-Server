"""Policy wiring and historical checkpoint compatibility for entity attention."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch

from Training.Mystic_Sim.Attention import EnemyAttentionEncoder
from Training.Mystic_Sim.core import PPOConfig, make_env, metadata, saved_config, load_checkpoint
from Training.Mystic_Sim.evaluate import load_evaluation_agent, evaluate
from Training.Mystic_Sim.recurrent import RecurrentAgent
from Training.Mystic_Sim.watch import PolicySession


class AttentionPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = make_env()
        self.addCleanup(self.env.close)

    def test_unpack_known_coordinates_padding_and_context(self):
        x = torch.zeros(2, 3, 80)
        x[..., :6] = .5
        x[..., 56:] = .25
        x[..., 8] = 1  # enemy 1 alive
        x[..., 28] = .5  # enemy 6 alive, no exact coordinates
        x[..., 46:48] = torch.tensor([-.2, .1])
        x[..., 48:50] = 1  # dead slot's garbage coordinates must not leak
        player, entities, valid, history = EnemyAttentionEncoder.unpack(x)
        self.assertEqual(tuple(entities.shape), (6, 10, 7))
        torch.testing.assert_close(entities[0, 0, 4:], torch.tensor([-.2, .1, 1.]))
        self.assertFalse(entities[:, 1].any())
        self.assertFalse(entities[:, 5, 4:].any())
        self.assertTrue(valid[:, 5].all())
        self.assertTrue((player == .5).all())
        self.assertTrue((history == .25).all())
        encoder = EnemyAttentionEncoder()
        whole = encoder(x)
        self.assertEqual(tuple(whole.shape), (2, 3, 512))
        torch.testing.assert_close(whole, torch.stack([encoder(t) for t in x]))

    def test_disjoint_parameters_and_gradients_through_attention(self):
        contract = metadata(self.env, recurrent=True)
        agent = RecurrentAgent(contract['observation_scale'])
        self.assertEqual(PPOConfig().recurrent_encoder, 'cross_attention')
        self.assertFalse({id(p) for p in agent.actor.parameters()} &
                         {id(p) for p in agent.critic.parameters()})
        obs = torch.rand(3, 2, 80) * agent.observation_scale
        logits, value, _, _ = agent.sequence(obs, agent.initial_state(2), torch.zeros(3, 2, dtype=torch.bool))
        (logits.square().sum() + value.square().sum()).backward()
        for branch in (agent.actor, agent.critic):
            for projection in ('W_q', 'W_k', 'W_v', 'W_o'):
                grad = getattr(branch.network.attention, projection).weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)

    def test_new_and_legacy_load_evaluate_and_watch(self):
        for encoder in ('flat', 'cross_attention'):
            config = PPOConfig(recurrent=True, recurrent_encoder=encoder)
            contract = metadata(self.env, recurrent=True, recurrent_encoder=encoder)
            original = RecurrentAgent(contract['observation_scale'], recurrent_encoder=encoder)
            settings = asdict(config)
            if encoder == 'flat':
                del settings['recurrent_encoder']  # historical checkpoint
                self.assertEqual(saved_config({'ppo_config': settings}).recurrent_encoder, 'flat')
            payload = dict(format='mystic_sim_ppo_v2', metadata=contract, ppo_config=settings,
                           agent=original.state_dict(), optimizer={}, update=1,
                           global_step=1024, rng={}, episodes=0, wins=0)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'model.pt'
                torch.save(payload, path)
                loaded, factory, _ = load_evaluation_agent(path)
                self.assertEqual(loaded.recurrent_encoder, encoder)
                for name, value in original.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[name], value)
                other = 'flat' if encoder == 'cross_attention' else 'cross_attention'
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    load_checkpoint(path, metadata(self.env, recurrent=True, recurrent_encoder=other))
                session = PolicySession(loaded, factory, seed=42, mode='greedy')
                try:
                    session.step()
                    session.reset()
                    self.assertTrue(session.episode_start)
                finally:
                    session.env.close()
                def short_factory():
                    env = factory()
                    env.config = replace(env.config, reward=replace(env.config.reward, max_episode_steps=2))
                    env.reward_config = env.config.reward
                    return env
                report = evaluate(loaded, [42], factory=short_factory)
                self.assertEqual(report['episodes'][0]['length'], 2)
