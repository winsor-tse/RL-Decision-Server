"""Checkpoint evaluation scheduling without training or rendering."""
from pathlib import Path
import unittest
from unittest.mock import patch

from Training.Mystic_Sim.curriculum_run import load_run_config, preflight
from Training.Mystic_Sim.train import checkpoint_recordings, full_evaluations


class EvaluationScheduleTests(unittest.TestCase):
    def test_gifs_start_at_checkpoint_five(self):
        directory=Path('run')
        for index in range(1,5):
            self.assertEqual(checkpoint_recordings(directory,index,True,5),(None,None))
        for index in range(5,11):
            paths=checkpoint_recordings(directory,index,True,5)
            self.assertEqual(paths[0],directory/'gameplay'/f'checkpoint_{index:02d}.gif')
            self.assertEqual(paths[1],directory/'gameplay'/f'checkpoint_{index:02d}_sampled.gif')
        self.assertEqual(checkpoint_recordings(directory,10,False,5),(None,None))

    def test_disabled_full_evaluation_does_not_collect_episodes(self):
        with patch('Training.Mystic_Sim.train.evaluate') as evaluate:
            reports=full_evaluations(None,[20000],enabled=False,encounter=object(),
                                    device='cpu',stage_reports={})
            self.assertEqual(reports,{})
            evaluate.assert_not_called()

    def test_hard_evaluates_both_modes_with_original_seeds(self):
        with patch('Training.Mystic_Sim.train.evaluate',return_value={'summary':{}}) as evaluate:
            reports=full_evaluations(None,[20000,20001],enabled=True,encounter=object(),
                                    device='cpu',stage_reports={'greedy':{},'sampled':{}})
            self.assertEqual(set(reports),{'greedy','sampled'})
            self.assertEqual(evaluate.call_count,2)
            self.assertEqual([c.kwargs['stochastic'] for c in evaluate.call_args_list],[False,True])
            self.assertTrue(all(c.args[1]==[20000,20001] for c in evaluate.call_args_list))

    def test_full_training_reuses_reports(self):
        with patch('Training.Mystic_Sim.train.evaluate') as evaluate:
            reports=full_evaluations(None,[20000],enabled=True,encounter=None,
                                    device='cpu',stage_reports={'greedy':{'value':1},'sampled':{'value':2}})
            self.assertEqual(reports['sampled']['value'],2)
            evaluate.assert_not_called()

    def test_pipeline_configs_and_preflight(self):
        for stage in ('easy','medium','hard'):
            run=load_run_config(f'Training/Mystic_Sim/configs/mixed_{stage}.json')
            self.assertEqual(run.ppo.eval_seed,20000)
            self.assertEqual(run.runtime['record_from_checkpoint'],5)
            self.assertEqual(run.runtime['evaluate_full_scenario'],stage=='hard')
            self.assertEqual(preflight(run)['full_scenario_evaluation'],stage=='hard')


if __name__=='__main__':
    unittest.main()
