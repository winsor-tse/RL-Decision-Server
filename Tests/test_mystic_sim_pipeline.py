"""Difficulty geometry, variable goals, and pipeline wiring; no training."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from Custom_enviornments.Mystic_Sim.encounters import reachable_distances
from Training.Mystic_Sim.curriculum import encounter_factory, validate_transfer_metadata
from Training.Mystic_Sim.curriculum_run import load_run_config
from Training.Mystic_Sim.core import metadata
from Training.Mystic_Sim.pipeline import build_pipeline, execute_pipeline

CONFIG = 'Training/Mystic_Sim/configs/'


class PipelineTests(unittest.TestCase):
    def test_medium_cp09_pipeline_source_and_handoff(self):
        with patch('Training.Mystic_Sim.pipeline.preflight'), patch(
                'Training.Mystic_Sim.pipeline.Path.exists',return_value=False):
            rows=build_pipeline(CONFIG+'medium_cp09_to_hard_pipeline.json')
            self.assertEqual([r['stage'] for r in rows],['medium_hard','hard_60_70_80'])
            self.assertTrue(rows[0]['command'][-1].endswith('checkpoint_09.pt'))
            self.assertEqual(rows[0]['command'][-2],'--continue-from')
            self.assertEqual(rows[1]['command'][-1],rows[0]['checkpoint'])
            overridden=build_pipeline(CONFIG+'medium_cp09_to_hard_pipeline.json',
                                      continue_from='override.pt')
            self.assertEqual(overridden[0]['command'][-1],str(Path('override.pt').resolve()))
        for filename,counts,weights in [
                ('mixed_medium_hard_from_medium_cp09.json',[40,50,60],[40,40,20]),
                ('mixed_hard_60_70_80.json',[60,70,80],[40,30,30])]:
            run=load_run_config(CONFIG+filename)
            self.assertEqual([c.encounter.enemy_count if c.encounter else 80 for c in run.encounter.mixture],counts)
            self.assertEqual([c.weight for c in run.encounter.mixture],weights)
            self.assertEqual(run.ppo.total_timesteps,1000000)
            self.assertEqual(run.ppo.learning_rate,.00025)

    def test_easy_nearby_geometry_and_goals_across_seeds(self):
        run=load_run_config(CONFIG+'mixed_easy.json')
        for choice in run.encounter.mixture:
            if choice.encounter.enemy_count > 3:
                continue
            env=encounter_factory(replace(run.encounter,mixture=(choice,)))()
            self.addCleanup(env.close)
            for seed in range(12):
                _,info=env.reset(seed=seed)
                player=env.world.player
                distances=reachable_distances(env.map_definition,(player.x,player.y))
                low,high=choice.encounter.path_distance
                self.assertLessEqual(high,9)
                self.assertEqual(len(env.world.monsters),choice.encounter.enemy_count)
                for npc in env.world.monsters.values():
                    self.assertTrue(low <= distances[npc.x,npc.y] <= high)
                    self.assertLessEqual(abs(npc.x-player.x),16)
                    self.assertLessEqual(abs(npc.y-player.y),10)
                goal=choice.encounter.kill_goal
                self.assertEqual(info['scenario_kill_goal'],goal)
                env.world.kills=goal-1
                self.assertFalse(env.step(4)[2])
                env.world.kills=goal
                self.assertTrue(env.step(4)[2])

    def test_sixty_covers_all_areas_and_retains_respawn_population(self):
        run=load_run_config(CONFIG+'mixed_hard.json')
        choice=next(c for c in run.encounter.mixture if c.name=='sixty_random')
        env=encounter_factory(choice.encounter)();self.addCleanup(env.close)
        results=[]
        for seed in (42,42,43):
            _,info=env.reset(seed=seed)
            counts=Counter(n.spawn_box.object_id for n in env.world.monsters.values())
            self.assertEqual(len(counts),40)
            self.assertEqual(Counter(counts.values()),{1:20,2:20})
            results.append(info['npc_spawns'])
        self.assertEqual(results[0],results[1])
        self.assertNotEqual(results[1],results[2])
        npc=next(iter(env.world.monsters.values()));box=npc.spawn_box
        env.engine.handle_death(npc,env.world.player.entity_id)
        env.world.time_ms=npc.respawn_at_ms
        env.engine.respawn(npc)
        self.assertEqual(npc.spawn_box,box)
        self.assertEqual(len(env.world.monsters),60)

    def test_stage_contracts_only_change_encounters(self):
        contracts=[]
        for stage in ('easy','medium','hard'):
            run=load_run_config(CONFIG+f'mixed_{stage}.json')
            env=encounter_factory(run.encounter)();self.addCleanup(env.close)
            contracts.append(metadata(env,recurrent=True))
            self.assertEqual(run.ppo.total_timesteps,1000000)
            self.assertEqual(run.ppo.learning_rate,.00025)
        validate_transfer_metadata(contracts[0],contracts[1])
        validate_transfer_metadata(contracts[1],contracts[2])

    def test_pipeline_commands_link_final_checkpoints(self):
        # Existing user training artifacts must not affect command-wiring tests.
        with patch('Training.Mystic_Sim.pipeline.Path.exists',return_value=False):
            rows=build_pipeline(CONFIG+'easy_medium_hard_pipeline.json')
        self.assertEqual([r['stage'] for r in rows],['easy','medium','hard'])
        self.assertIn('--fresh',rows[0]['command'])
        for old,new in zip(rows,rows[1:]):
            self.assertEqual(new['command'][-2:],['--continue-from',old['checkpoint']])
        with self.assertRaises(ValueError):
            build_pipeline(CONFIG+'easy_medium_hard_pipeline.json',start_stage='medium')

    def test_runner_stops_on_failure_or_incomplete_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            row={'stage':'easy','transitions':100,'command':['mock'],
                 'run_dir':folder,'checkpoint':str(Path(folder)/'checkpoint_10.pt')}
            Path(folder,'progress.json').write_text(json.dumps({'complete':False,'global_step':50}))
            with patch('Training.Mystic_Sim.pipeline.subprocess.run') as launch:
                with self.assertRaises(RuntimeError): execute_pipeline([row,row])
                self.assertEqual(launch.call_count,1)
            with patch('Training.Mystic_Sim.pipeline.subprocess.run',
                       side_effect=subprocess.CalledProcessError(1,['mock'])) as launch:
                with self.assertRaises(subprocess.CalledProcessError): execute_pipeline([row,row])
                self.assertEqual(launch.call_count,1)
            Path(row['checkpoint']).touch()
            Path(folder,'progress.json').write_text(json.dumps({'complete':True,'global_step':100}))
            with patch('Training.Mystic_Sim.pipeline.subprocess.run') as launch:
                execute_pipeline([row,row])
                self.assertEqual(launch.call_count,2)

    def test_requested_random_population_mixtures(self):
        for stage,populations in [('medium',[20,30,40,50]),('hard',[50,60,70,80])]:
            run=load_run_config(CONFIG+f'mixed_{stage}.json')
            self.assertEqual([c.encounter.enemy_count if c.encounter else 80
                              for c in run.encounter.mixture],populations)
            weights=[c.weight for c in run.encounter.mixture]
            self.assertEqual([w/sum(weights) for w in weights],
                             [.3,.3,.3,.1] if stage=='medium' else [.25]*4)
            for choice,count in zip(run.encounter.mixture,populations):
                env=encounter_factory(choice.encounter)();self.addCleanup(env.close)
                _,info=env.reset(seed=42)
                counts=Counter(n.spawn_box.object_id for n in env.world.monsters.values())
                self.assertEqual(sum(counts.values()),count)
                self.assertEqual(len(counts),min(count,40))
                self.assertEqual(sum(v==2 for v in counts.values()),max(0,count-40))
                self.assertEqual(env.reward_config.win_kills,5)
                if choice.encounter and count<=40:
                    self.assertEqual(choice.encounter.placement,'random_spawn_areas_single_v1')


if __name__=='__main__':
    unittest.main()
