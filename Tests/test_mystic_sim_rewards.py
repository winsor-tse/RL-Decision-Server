"""Reward contracts, Gym episode boundaries, and diagnostic isolation."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from Custom_enviornments.Mystic_Sim.config import ScenarioConfig, RewardConfig
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.engine import Engine
from Custom_enviornments.Mystic_Sim.observation import encode_observation
from Custom_enviornments.Mystic_Sim import rewards


class RewardTests(unittest.TestCase):
    def scene(self, reward=None):
        env = MysticSimEnv(config=ScenarioConfig(reward=reward or RewardConfig()), trace=True)
        self.addCleanup(env.close)
        env.reset(seed=42)
        w = env.world
        w.player.x, w.player.y = 50, 50
        w.monsters = {2: w.monsters[2]}
        w.monsters[2].x, w.monsters[2].y = 55, 50
        w.events.clear()
        w.occupancy = {(50,50):1, (55,50):2}
        env.engine = Engine(w, env.config, env.np_random, trace=True)
        env.engine.queue_npc = lambda *args, **kwargs: None
        return env

    def transition(self, env, callback):
        original = env.engine.dispatch
        env.engine.scheduler.schedule(env.world.time_ms+100, 'test_event', 1)
        with patch.object(env.engine, 'dispatch', side_effect=lambda e: callback() if e.kind == 'test_event' else original(e)):
            return env.step(4)

    def test_combat_event_damage_not_masked_by_regeneration(self):
        env = self.scene()
        p, npc = env.world.player, env.world.monsters[2]
        def damage_then_heal():
            env.engine.apply_damage(npc, p, 100)
            env.engine.regenerate()
            env.engine.apply_damage(p, npc, 1000, damage_type='dot')
        obs, reward, term, trunc, info = self.transition(env, damage_then_heal)
        self.assertEqual(obs[3], 1)
        parts = info['reward_components']
        self.assertAlmostEqual(parts['damage_taken'], -100/p.max_hp)
        self.assertAlmostEqual(parts['damage_dealt'], 1000/npc.max_hp)
        self.assertEqual(parts['health_state'], -.001)
        self.assertEqual(reward, sum(parts.values()))
        self.assertEqual((term,trunc),(False,False))
        self.assertEqual(set(parts), set(rewards.COMPONENTS))

    def test_overkill_kill_bonus_and_no_reward_for_disappearance(self):
        env = self.scene()
        p,npc = env.world.player,env.world.monsters[2]
        npc.hp = 100
        result = self.transition(env,lambda: env.engine.apply_damage(p,npc,1000000))
        parts = result[4]['reward_components']
        self.assertAlmostEqual(parts['damage_dealt'],100/npc.max_hp)
        self.assertEqual(parts['killed'],1)
        next_result = env.step(4)
        self.assertEqual(next_result[4]['reward_components']['killed'],0)
        env = self.scene()
        def disappear():
            env.world.monsters.clear()
            env.world.occupancy.pop((55,50))
        result = self.transition(env, disappear)
        self.assertEqual(result[4]['reward_components']['damage_dealt'],0)
        self.assertEqual(result[4]['reward_components']['killed'],0)

    def test_terminal_win_death_and_time_limit_priority(self):
        for loss in (False,True):
            env = self.scene()
            w = env.world
            w.step_count = 255
            w.kills = 4
            def finish():
                env.engine.apply_damage(w.player,w.monsters[2],w.monsters[2].hp)
                if loss: env.engine.apply_damage(w.monsters[2],w.player,w.player.hp)
            result = self.transition(env,finish)
            self.assertEqual(result[2:4],(True,False))
            self.assertEqual(result[4]['episode_outcome'],'loss' if loss else 'win')
            self.assertEqual(result[4]['reward_components']['terminal'],-5 if loss else 0)
            with self.assertRaises(gym.error.ResetNeeded): env.step(4)
        env = self.scene()
        env.world.step_count = 255
        result=env.step(4)
        self.assertEqual(result[2:4],(False,True))
        self.assertEqual(result[4]['episode_end_reason'],'step_limit')
        self.assertEqual(result[4]['reward_components']['terminal'],0)

    def test_y_rules_can_be_disabled_or_overridden(self):
        env=self.scene(RewardConfig(y_bounds=None))
        env.world.player.y=0
        self.assertEqual(env.step(4)[2:4],(False,False))
        for y, outside in ((24,True),(25,False),(60,False),(61,True)):
            env=self.scene(RewardConfig(y_bounds=(25,60)))
            env.world.player.y=y
            result=env.step(4)
            self.assertEqual(result[2:4],(False,outside))
            if outside: self.assertEqual(result[4]['episode_end_reason'],'y_boundary')

    def test_legacy_numeric_snapshot_and_healing_quirk(self):
        env=self.scene(RewardConfig(profile='legacy_reward_v0',legacy_y_penalty_below=31))
        w=env.world
        prev=np.zeros(26,dtype=np.float32);prev[:7]=[50,30,0,.75,1,53,5]
        obs=prev.copy();obs[3]=.125
        parts=rewards.calculate(w,env.reward_config,prev,obs,0,[],[],'loss')
        self.assertEqual(parts,dict(health_state=-.5,positioning=-107.,damage_taken=-12.5,
                                    damage_dealt=0.,terminal=-100.,killed=0.))
        prev[3]=.25;obs[3]=.75
        self.assertEqual(rewards.calculate(w,env.reward_config,prev,obs,4,[],[],None)['damage_taken'],10)
        obs[3]=.5
        self.assertEqual(rewards.calculate(w,env.reward_config,prev,obs,4,[],[],None)['damage_taken'],0)
        combat=RewardConfig()
        obs[3]=.75
        self.assertEqual(rewards.calculate(w,combat,prev,obs,4,[],[],None)['damage_taken'],0)

    def test_legacy_enemy_events_and_optional_y_penalty(self):
        env=self.scene(RewardConfig(profile='legacy_reward_v0'))
        p,npc=env.world.player,env.world.monsters[2]
        result=self.transition(env,lambda: env.engine.apply_damage(p,npc,npc.hp))
        self.assertEqual(result[4]['reward_components']['damage_dealt'],25)
        self.assertEqual(result[4]['reward_components']['killed'],10)
        env=self.scene(RewardConfig(profile='legacy_reward_v0'))
        obs=encode_observation(env.world);obs[1]=0
        parts=rewards.calculate(env.world,env.reward_config,obs,obs,4,[],[],None)
        self.assertEqual(parts['positioning'],-3097)  # +3 shaping, -3100 Y penalty.

    def test_y_penalty_enabled_for_both_profiles(self):
        for profile in ('combat_reward_v1','legacy_reward_v0'):
            env=self.scene(RewardConfig(profile=profile))
            obs=encode_observation(env.world)
            for y,penalty in ((31,0),(30,-100),(29,-200)):
                obs[1]=y
                parts=rewards.calculate(env.world,env.reward_config,obs,obs,4,[],[],None)
                self.assertEqual(parts['positioning'],penalty+(3 if profile=='legacy_reward_v0' else 0))
            config=replace(env.reward_config,legacy_y_penalty_below=None)
            self.assertEqual(rewards.calculate(env.world,config,obs,obs,4,[],[],None)['positioning'],
                             3 if profile=='legacy_reward_v0' else 0)

    def test_sim_y_penalty_and_truncation_boundaries(self):
        for profile in ('combat_reward_v1','legacy_reward_v0'):
            for y,penalty,truncated in ((28,-300,True),(29,-200,True),(30,-100,False),
                                        (31,0,False),(85,0,False),(86,-100,False),
                                        (87,-200,True),(88,-300,True)):
                with self.subTest(profile=profile,y=y):
                    env=self.scene(RewardConfig(profile=profile))
                    env.world.player.y=y
                    env.world.monsters[2].y=y  # Keep legacy distance shaping constant at +3.
                    obs,reward,term,trunc,info=env.step(4)
                    self.assertEqual((term,trunc),(False,truncated))
                    self.assertEqual(info['reward_components']['positioning'],
                                     penalty+(3 if profile=='legacy_reward_v0' else 0))
                    self.assertEqual(reward,sum(info['reward_components'].values()))
                    if truncated:
                        self.assertEqual(info['episode_end_reason'],'y_boundary')
                        self.assertEqual(info['reward_components']['terminal'],0)
                        with self.assertRaises(gym.error.ResetNeeded): env.step(4)

    def test_reset_profiles_are_local_and_validation_is_atomic(self):
        a,b=self.scene(),self.scene()
        obs,info=a.reset(seed=7,options={'profile':a.config.profile,'reward_profile':'legacy_reward_v0'})
        self.assertEqual(info['reward_profile'],'legacy_reward_v0')
        self.assertEqual(a.config.reward.profile,'combat_reward_v1')
        self.assertEqual(b.reward_config.profile,'combat_reward_v1')
        before=deepcopy(a.world)
        for options in ({'reward_profile':'invalid'},{'profile':'invalid'},{'unknown':1}):
            with self.assertRaises(ValueError): a.reset(seed=99,options=options)
            self.assertEqual(a.world,before)
        _,info=a.reset(seed=7)
        self.assertEqual(info['reward_profile'],'combat_reward_v1')
        self.assertEqual(set(info['reward_components']),set(rewards.COMPONENTS))

    def test_diagnostics_complete_and_detached(self):
        env=self.scene()
        obs,reward,term,trunc,info=env.step(6)
        for key in ('profile','reward_profile','seed','simulation_time_ms','current_step','selected_target',
                    'action_applied','action_failure_reason','reward_components','kills','cooldowns',
                    'active_effects','damage_events','death_events','respawn_events'):
            self.assertIn(key,info)
        self.assertEqual(info['selected_target'],2)
        self.assertEqual(info['seed'],42)
        info['cooldowns']['slots'].clear()
        info['active_effects'][0]['tick_damage']=-1
        info['trace'][0]['kind']='tampered'
        self.assertEqual(env.world.player.cooldowns.slots[2],5000)
        self.assertGreater(env.world.effects[0].tick_damage,0)
        self.assertNotEqual(env.engine.trace[0]['kind'],'tampered')
        self.assertTrue(info['reward_implemented'])

    def test_gym_checker_bounds_and_invalid_action_no_mutation(self):
        for profile in ('combat_reward_v1','legacy_reward_v0'):
            env=MysticSimEnv(config=ScenarioConfig(reward=RewardConfig(profile=profile)))
            self.addCleanup(env.close)
            check_env(env,skip_render_check=True)
            obs,info=env.reset(seed=8)
            self.assertEqual(env.action_space.n,8)
            self.assertEqual(obs.shape,(26,))
            self.assertEqual(obs.dtype,np.float32)
            before=deepcopy(env.world)
            for action in (-1,8,True,1.5):
                with self.assertRaises(ValueError):env.step(action)
                self.assertEqual(env.world,before)
            for i in range(80):
                obs,reward,term,trunc,info=env.step(i%8)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertTrue(np.isfinite(reward))
                self.assertEqual(reward,sum(info['reward_components'].values()))
                if term or trunc: break

    def test_seeded_profiles_replay_byte_equal(self):
        for profile in ('combat_reward_v1','legacy_reward_v0'):
            a,b=MysticSimEnv(trace=True),MysticSimEnv(trace=True)
            self.addCleanup(a.close);self.addCleanup(b.close)
            ra=a.reset(seed=123,options={'reward_profile':profile})
            rb=b.reset(seed=123,options={'reward_profile':profile})
            self.assertEqual(ra[0].tobytes(),rb[0].tobytes())
            self.assertEqual(ra[1],rb[1])
            for i in range(100):
                ra,rb=a.step(i%8),b.step(i%8)
                self.assertEqual(ra[0].tobytes(),rb[0].tobytes())
                self.assertEqual(ra[1:],rb[1:])
                if ra[2] or ra[3]: break


if __name__ == '__main__':
    unittest.main()
