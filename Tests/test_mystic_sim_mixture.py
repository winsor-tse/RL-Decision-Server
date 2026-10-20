"""Fixed mixture contracts and resets; no PPO training."""
from dataclasses import asdict, replace
import io
import unittest
import numpy as np
import torch

from Training.Mystic_Sim.core import SimBatch, metadata
from Training.Mystic_Sim.curriculum import encounter_factory, encounter_from_dict, checkpoint_factory
from Training.Mystic_Sim.curriculum_run import load_run_config, preflight
from Training.Mystic_Sim.mixed_scenarios import ScenarioChoice


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.run = load_run_config('Training/Mystic_Sim/configs/mixed_easy_medium_hard.json')

    def env(self, encounter=None):
        env = encounter_factory(encounter or self.run.encounter)()
        self.addCleanup(env.close)
        return env

    def test_fresh_defaults_and_preflight(self):
        self.assertIsNone(self.run.continue_from)
        self.assertTrue(self.run.ppo.recurrent)
        self.assertEqual(self.run.ppo.learning_rate, .00025)
        report = preflight(self.run)
        self.assertFalse(report['automatic_advancement'])
        self.assertTrue(report['full_scenario_evaluation'])

    def test_roundtrip_checkpoint_metadata_and_seed(self):
        env = self.env()
        contract = metadata(env, recurrent=True)
        restored = checkpoint_factory({'metadata': contract})()
        self.addCleanup(restored.close)
        self.assertEqual(self.run.encounter, encounter_from_dict(asdict(self.run.encounter)))
        for seed in (42, 43, 44, 27000):
            a, ai = env.reset(seed=seed)
            b, bi = restored.reset(seed=seed)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(ai['training_scenario'],bi['training_scenario'])
            self.assertEqual(ai['npc_spawns'],bi['npc_spawns'])
        self.assertEqual(metadata(env,recurrent=True),contract)

    def test_all_members_keep_stats_goal_and_original_map(self):
        for choice, count in zip(self.run.encounter.mixture,(5,20,40,80)):
            env = self.env(replace(self.run.encounter,mixture=(choice,)))
            obs, info = env.reset(seed=42)
            self.assertEqual(len(env.world.monsters),count)
            self.assertEqual(env.active.config.reward,env.config.reward)
            self.assertEqual(env.active.config.innie,env.config.innie)
            self.assertEqual(env.active.map_definition.map_id,53)
            self.assertTrue(env.observation_space.contains(obs))
            active = env.active
            for _ in range(3):
                result = env.step(4)
                self.assertEqual(result[4]['training_scenario'],choice.name)
                self.assertIs(env.active,active)

    def test_sampler_state_safe_serialization_and_continuation(self):
        env, other = self.env(), self.env()
        env.reset(seed=42)
        stream=io.BytesIO()
        torch.save(env.sampling_state(),stream);stream.seek(0)
        other.restore_sampling_state(torch.load(stream,weights_only=True))
        for _ in range(6):
            a, ai=env.reset();b, bi=other.reset()
            self.assertEqual(ai['training_scenario'],bi['training_scenario'])
            self.assertEqual(ai['scenario_seed'],bi['scenario_seed'])
            np.testing.assert_array_equal(a,b)

    def test_batch_terminal_info_and_history_reset(self):
        batch=SimBatch(2,42,factory=encounter_factory(self.run.encounter),recurrent=True)
        self.addCleanup(batch.close)
        name=batch.envs[0].choice.name
        # Force the existing time limit to test autoreset, not an optimizer run.
        batch.envs[0].world.step_count=1023
        final,rewards,term,trunc,infos=batch.step([4,4])
        self.assertEqual(infos[0]['training_scenario'],name)
        self.assertTrue(trunc[0])
        self.assertEqual(batch.envs[0].world.step_count,0)
        np.testing.assert_array_equal(batch.histories[0].features,np.zeros(16))
        self.assertEqual(final.shape,batch.observations.shape)
        self.assertEqual(batch.envs[1].world.step_count,1)

    def test_independent_world_rng_and_batch_sampler_restore(self):
        batch=SimBatch(2,42,factory=encounter_factory(self.run.encounter),recurrent=True)
        self.addCleanup(batch.close)
        states=[e.sampling_state() for e in batch.envs]
        batch.envs[0].active.np_random.random(100)
        expected=[e.reset()[0] for e in batch.envs]
        restored=SimBatch(2,999,factory=encounter_factory(self.run.encounter),sampling_states=states)
        self.addCleanup(restored.close)
        np.testing.assert_array_equal(np.stack(expected),restored.observations)
        self.assertIsNot(batch.envs[0].world,batch.envs[1].world)

    def test_invalid_mixtures(self):
        base=self.run.encounter
        first=base.mixture[0]
        for weight in (0,-1,float('inf'),float('nan'),True):
            with self.assertRaises(ValueError):
                replace(base,mixture=(replace(first,weight=weight),))
        for choices in ((),(first,first),
                        (replace(first,name='../invalid'),),
                        (ScenarioChoice('nested',1,base),)):
            with self.assertRaises(ValueError):
                replace(base,mixture=choices)

    def test_evaluation_reports_selected_member(self):
        from Training.Mystic_Sim.evaluate import evaluate
        report=evaluate(None,[20000],factory=encounter_factory(self.run.encounter))
        self.assertEqual(len(report['episodes']),1)
        self.assertIn(report['episodes'][0]['training_scenario'],
                      {c.name for c in self.run.encounter.mixture})

    def test_episode_probabilities_and_all_scenarios_reachable(self):
        from collections import Counter
        env=self.env()
        np.testing.assert_allclose(env.probabilities,[.2,.5,.2,.1])
        env.reset(seed=123)
        counts=Counter(env.reset()[1]['training_scenario'] for _ in range(200))
        self.assertEqual(set(counts),{'easy','medium','hard','full'})
        for choice,p in zip(env.mixture.mixture,env.probabilities):
            self.assertLess(abs(counts[choice.name]/200-p),.1)


if __name__ == '__main__':
    unittest.main()
