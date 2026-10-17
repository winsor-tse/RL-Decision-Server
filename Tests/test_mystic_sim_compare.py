"""Comparison wiring tests; no training or optimizer updates."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Training.Mystic_Sim import compare
from Training.Mystic_Sim.evaluate import evaluation_line


CONFIG = Path('Training/Mystic_Sim/configs/compare_4v1_5v1.json')


class ComparisonTests(unittest.TestCase):
    def test_config_resolves_ten_enemies_with_five_kill_goal(self):
        data, runs = compare.load_comparison(CONFIG)
        encounter = runs['ten_innies_goal5'].encounter
        self.assertEqual((encounter.enemy_count, encounter.kill_goal), (10, 5))
        self.assertIsNone(runs['full'].encounter)
        self.assertEqual(data['modes'], ['sampled', 'greedy'])

    def test_pairings_share_seeds_and_select_both_modes(self):
        report = {'summary': {'episode_return': 100, 'win': .5, 'kills': 3, 'length': 40}}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(compare, 'load_evaluation_agent', return_value=(object(), None, {})), \
                patch.object(compare, 'validate_transfer_metadata'), \
                patch.object(compare, 'evaluate', return_value=report) as evaluate, \
                redirect_stdout(io.StringIO()) as stdout:
            result = compare.main(['--config', str(CONFIG), '--episodes', '2', '--seed', '23000',
                                   '--output', str(Path(temp) / 'result.json')])
            self.assertEqual(evaluate.call_count, 8)
            for call in evaluate.call_args_list:
                self.assertEqual(call.args[1], [23000, 23001])
            self.assertEqual([c.kwargs['stochastic'] for c in evaluate.call_args_list],
                             [True, False] * 4)
            self.assertFalse(result['training_performed'])
            self.assertIn('4v1_cp10 / ten_innies_goal5 / sampled:', stdout.getvalue())
            for label in result['results']:
                self.assertIn(f'{label} / full / greedy:', stdout.getvalue())

    def test_invalid_contract_fails_before_rollouts(self):
        with patch.object(compare, 'load_evaluation_agent', return_value=(object(), None, {})), \
                patch.object(compare, 'validate_transfer_metadata', side_effect=ValueError('incompatible')), \
                patch.object(compare, 'evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                compare.main(['--config', str(CONFIG), '--validate-only'])
            evaluate.assert_not_called()

    def test_summary_labels_and_metrics(self):
        line = evaluation_line('stage sampled', {'summary': {
            'episode_return': 12.5, 'win': .8, 'kills': 4, 'length': 500}})
        self.assertEqual(line, 'stage sampled: return=12.50, win=80.0%, kills=4.00, steps=500.0')


if __name__ == '__main__':
    unittest.main()
