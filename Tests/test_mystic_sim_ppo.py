"""Standalone PPO numerical targets, real training, resume, and recording gates."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.config import ScenarioConfig
from Training.Mystic_Sim.core import (Agent, PPOConfig, SimBatch, advantages,
    checkpoint_updates, load_checkpoint, make_env, metadata)
from Training.Mystic_Sim.evaluate import evaluate
from Training.Mystic_Sim.train import train, critic_loss, clip_gradients
from Utils.ppo_checkpoint import capture_rng_state


def short_env():
    config = ScenarioConfig()
    return MysticSimEnv(config=replace(config,reward=replace(config.reward,max_episode_steps=8)))


class MysticPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gae_terminal_truncated_and_cross_episode_masks(self):
        # Transition 0 truncates; transition 1 belongs to a new, terminal episode.
        reward=torch.tensor([[1.,1.,1.],[100.,100.,100.]])
        value=torch.zeros_like(reward)
        next_value=torch.tensor([[10.,10.,10.],[999.,999.,999.]])
        term=torch.tensor([[True,False,False],[True,True,True]])
        trunc=torch.tensor([[False,True,False],[False,False,False]])
        adv,returns=advantages(reward,value,next_value,term,trunc,.9,1.)
        torch.testing.assert_close(adv,torch.tensor([[1.,10.,100.],[100.,100.,100.]]))
        torch.testing.assert_close(returns,adv)

    def test_batch_preserves_terminal_observation_and_independent_resets(self):
        batch=SimBatch(2,7,factory=short_env)
        self.addCleanup(batch.close)
        a,b=batch.envs
        self.assertIsNot(a.world,b.world)
        self.assertIsNot(a.np_random,b.np_random)
        a.world.events.clear();b.world.events.clear()
        a.world.step_count=7
        a.world.player.hp=123
        final,rewards,term,trunc,infos=batch.step([4,4])
        self.assertEqual(trunc.tolist(),[True,False])
        self.assertEqual(term.tolist(),[False,False])
        self.assertAlmostEqual(float(final[0,3]),123/a.world.player.max_hp)
        self.assertEqual(batch.observations[0,3],1)
        self.assertEqual(infos[0]['current_step'],8)
        self.assertEqual(a.world.step_count,0)
        self.assertEqual(b.world.step_count,1)

    def test_checkpoint_schedule_and_invalid_configs(self):
        self.assertEqual(checkpoint_updates(10),{i:i for i in range(1,11)})
        self.assertEqual(len(checkpoint_updates(103)),10)
        self.assertEqual(max(checkpoint_updates(103)),103)
        for config in (PPOConfig(num_envs=0),PPOConfig(total_timesteps=1),
                       PPOConfig(num_minibatches=7),PPOConfig(gamma=2)):
            with self.assertRaises(ValueError): config.validate()

    def test_scaled_training_defaults_and_validation(self):
        config=PPOConfig()
        self.assertEqual(config.reward_scale,.01)
        self.assertEqual(config.gamma,.999)
        self.assertIsNone(config.value_clip_coef)
        for name in ('reward_scale','actor_max_grad_norm','critic_max_grad_norm','value_clip_coef'):
            for value in (0,-1,float('nan'),float('inf')):
                with self.assertRaises(ValueError): replace(config,**{name:value}).validate()
        replace(config,value_clip_coef=.3).validate()

    def test_value_clip_is_optional_and_in_value_units(self):
        value=torch.tensor([2.],requires_grad=True)
        old=torch.tensor([0.]);target=torch.tensor([10.])
        loss=critic_loss(value,old,target)
        self.assertEqual(float(loss.detach()),32.)
        loss.backward()
        self.assertEqual(float(value.grad),-8.)
        self.assertAlmostEqual(float(critic_loss(value,old,target,.2).detach()),48.02,places=4)
        self.assertEqual(float(critic_loss(value,old,target,3.).detach()),32.)

    def test_huge_critic_gradient_does_not_shrink_actor(self):
        from types import SimpleNamespace
        agent=SimpleNamespace(actor=torch.nn.Linear(1,1,bias=False),
                              critic=torch.nn.Linear(1,1,bias=False))
        agent.actor.weight.grad=torch.tensor([[.2]])
        agent.critic.weight.grad=torch.tensor([[1000.]])
        actor,critic=clip_gradients(agent,PPOConfig())
        self.assertAlmostEqual(float(actor),.2)
        self.assertEqual(float(critic),1000.)
        self.assertAlmostEqual(float(agent.actor.weight.grad),.2)
        self.assertAlmostEqual(float(agent.critic.weight.grad),.5,places=5)

    def test_cpu_training_ten_checkpoints_resume_metadata_and_tensorboard(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            config=PPOConfig(total_timesteps=80,num_steps=8,num_minibatches=2,
                             update_epochs=1,eval_episodes=1,seed=19)
            real_evaluate=evaluate
            def quick_evaluation(*args,**kwargs):
                return real_evaluate(*args,**kwargs,factory=short_env)
            with patch('Training.Mystic_Sim.train.evaluate',side_effect=quick_evaluation):
                train(config,run_dir=directory,device='cpu',record_gameplay=False,stop_after_checkpoint=1)
                first=directory/'checkpoints/checkpoint_01.pt'
                original_bytes=first.read_bytes()
                env=make_env()
                expected=metadata(env)
                env.close()
                saved=load_checkpoint(first,expected)
                self.assertTrue(saved['optimizer']['state'])
                self.assertEqual(saved['global_step'],8)
                self.assertEqual(saved['update'],1)
                torch.manual_seed(config.seed)
                agent=Agent(expected['observation_scale'])
                self.assertFalse(torch.equal(agent.actor[0].weight,saved['agent']['actor.0.weight']))
                # Resume must restore counters, optimizer, hyperparameters and RNG.
                from Training.Mystic_Sim.train import restore_rng_state as real_restore
                with patch('Training.Mystic_Sim.train.restore_rng_state',wraps=real_restore) as restore:
                    train(PPOConfig(),run_dir=directory,device='cpu',resume=first,record_gameplay=False)
                    self.assertEqual(restore.call_count,1)
                    torch.testing.assert_close(restore.call_args.args[0]['torch'],saved['rng']['torch'])
                self.assertEqual(first.read_bytes(),original_bytes)
                files=list((directory/'checkpoints').glob('*.pt'))
                self.assertEqual(len(files),10)
                last=load_checkpoint(directory/'checkpoints/checkpoint_10.pt',expected)
                self.assertEqual(last['global_step'],80)
                self.assertEqual(last['ppo_config'],saved['ppo_config'])
                self.assertEqual(last['ppo_config']['reward_scale'],.01)
                self.assertEqual(last['ppo_config']['gamma'],.999)
                self.assertIsNone(last['ppo_config']['value_clip_coef'])
                self.assertGreater(next(iter(last['optimizer']['state'].values()))['step'],
                                   next(iter(saved['optimizer']['state'].values()))['step'])
                progress=json.loads((directory/'progress.json').read_text())
                self.assertTrue(progress['complete'])
                self.assertGreater(progress['parameter_delta'],0)
                self.assertTrue(all(np.isfinite(list(progress['losses'].values()))))
                events=EventAccumulator(str(directory/'tensorboard')).Reload()
                for tag in ('losses/value_loss','charts/SPS_training','rewards/killed',
                            'rewards/damage_dealt','evaluation/return_vs_random','charts/parameter_delta',
                            'losses/actor_grad_norm','losses/critic_grad_norm','losses/critic_hidden_saturation'):
                    self.assertIn(tag,events.Tags()['scalars'])
                raw=events.Scalars('charts/mean_step_reward')
                scaled=events.Scalars('charts/mean_scaled_step_reward')
                for r,s in zip(raw,scaled):
                    self.assertEqual(r.step,s.step)
                    self.assertAlmostEqual(s.value,r.value*.01,places=5)
                self.assertEqual(saved['format'],'mystic_sim_ppo_v2')
                old_path=directory/'old_version.pt'
                torch.save({**saved,'format':'mystic_sim_ppo_v1'},old_path)
                with self.assertRaisesRegex(ValueError,'start a new run'):
                    load_checkpoint(old_path,expected)
                self.assertAlmostEqual(events.Scalars('charts/learning_rate')[-1].value,
                                       config.learning_rate/10,places=8)
                incompatible=deepcopy(expected)
                incompatible['contract']['actions'][0]='wrong'
                with self.assertRaises(ValueError): load_checkpoint(first,incompatible)
                legacy=directory/'legacy.pt';torch.save(agent.state_dict(),legacy)
                with self.assertRaises(ValueError): load_checkpoint(legacy,expected)

    def test_evaluation_reload_determinism_and_rng_isolation(self):
        env=make_env()
        self.addCleanup(env.close)
        torch.manual_seed(4)
        agent=Agent(env.observation_space.high)
        before=capture_rng_state()
        a=evaluate(agent,[123,124],stochastic=True,factory=short_env)
        b=evaluate(agent,[123,124],stochastic=True,factory=short_env)
        self.assertEqual(a['episodes'],b['episodes'])
        after=capture_rng_state()
        torch.testing.assert_close(before['torch'],after['torch'])
        self.assertEqual(before['python'],after['python'])
        torch.testing.assert_close(before['numpy']['keys'],after['numpy']['keys'])
        self.assertEqual(before['numpy']['position'],after['numpy']['position'])
        clone=Agent(env.observation_space.high)
        clone.load_state_dict(agent.state_dict())
        observation=torch.tensor(env.reset(seed=1)[0]).unsqueeze(0)
        torch.testing.assert_close(agent.greedy(observation),clone.greedy(observation))

    @unittest.skipUnless(importlib.util.find_spec('pygame') and importlib.util.find_spec('PIL'),
                         'Optional recording packages unavailable')
    def test_pygame_gif_recording_and_missing_recorder_fallback(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gameplay.gif'
            result=evaluate(None,[3],record_path=path,record_stride=2,factory=short_env)
            self.assertIsNone(result['recording_error'])
            with Image.open(path) as image:
                self.assertEqual(image.size,(660,430))
                self.assertGreater(image.n_frames,1)
            with patch('Training.Mystic_Sim.evaluate.GameplayRecorder',side_effect=ImportError('missing pygame')):
                result=evaluate(None,[3],record_path=Path(directory)/'unavailable.gif',factory=short_env)
            self.assertIn('missing pygame',result['recording_error'])
            self.assertEqual(len(result['episodes']),1)


if __name__ == '__main__':
    unittest.main()
