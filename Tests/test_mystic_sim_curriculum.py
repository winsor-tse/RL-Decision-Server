"""Curriculum contract/unit checks only: no training or optimizer steps."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from Custom_enviornments.Mystic_Sim.config import EncounterConfig
from Training.Mystic_Sim.core import PPOConfig, metadata
from Training.Mystic_Sim.curriculum import (
    encounter_factory, checkpoint_factory, learning_rate, load_stage_checkpoint,
    stage_schedule, validate_transfer_metadata,
)
from Training.Mystic_Sim.curriculum_run import load_run_config, main, preflight
from Training.Mystic_Sim.recurrent import RecurrentAgent
from Utils.ppo_checkpoint import capture_rng_state


CONFIG = Path('Training/Mystic_Sim/configs/curriculum_manual.json')


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def write_config(self, modify=None):
        data = json.loads(CONFIG.read_text())
        if modify:
            modify(data)
        path = self.path / 'config.json'
        path.write_text(json.dumps(data))
        return path

    def checkpoint(self):
        run = load_run_config(CONFIG)
        env = encounter_factory(run.encounter)()
        self.addCleanup(env.close)
        contract = metadata(env, recurrent=True, actor_head='linear')
        torch.set_num_threads(1)
        agent = RecurrentAgent(contract['observation_scale'], 'linear')
        optimizer = torch.optim.Adam(agent.parameters(), lr=run.ppo.learning_rate, eps=1e-5)
        # Synthetic distinct learned values/Adam moments; never call backward/step.
        with torch.no_grad():
            for index, parameter in enumerate(agent.parameters()):
                parameter.fill_((index + 1) / 100)
                optimizer.state[parameter] = {
                    'step': torch.tensor(7.), 'exp_avg': torch.full_like(parameter, .125),
                    'exp_avg_sq': torch.full_like(parameter, .25)}
        payload = {'format': 'mystic_sim_ppo_v2', 'metadata': contract,
                   'ppo_config': asdict(run.ppo), 'agent': agent.state_dict(),
                   'optimizer': optimizer.state_dict(), 'update': run.ppo.updates,
                   'global_step': run.ppo.updates * run.ppo.batch_size,
                   'episodes': 0, 'wins': 0, 'rng': capture_rng_state(),
                   'curriculum': stage_schedule(run.ppo, schedule_timesteps=run.schedule_timesteps)}
        path = self.path / 'checkpoint.pt'
        torch.save(payload, path)
        return path, payload, run

    def test_all_configured_stages_are_feasible_without_steps(self):
        data = json.loads(CONFIG.read_text())
        for stage in data['stages']:
            with self.subTest(stage=stage):
                run = load_run_config(CONFIG, stage)
                report = preflight(run)
                self.assertEqual(report['stage'], stage)
                self.assertFalse(report['automatic_advancement'])
                self.assertEqual(report['local_timesteps'], 200704)
                self.assertEqual(len(report['evaluation_seeds']), 30)
                if stage == 'full':
                    self.assertIsNone(run.encounter)

    def test_invalid_configuration_and_seed_overlap(self):
        changes = [lambda d: d.update(unknown=True),
                   lambda d: d['ppo'].update(recurrent='false'),
                   lambda d: d['ppo'].update(num_envs=True),
                   lambda d: d['runtime'].update(record_gameplay='yes'),
                   lambda d: d['stages']['one_innie'].update(path_distance=None),
                   lambda d: d.update(stage='missing'),
                   lambda d: d['stages'].update(easy=None),
                   lambda d: d.update(schedule_timesteps=-1)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_run_config(self.write_config(change))
        run = load_run_config(self.write_config(lambda d: d['ppo'].update(seed=20000)))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            preflight(run)

    def test_transfer_preserves_all_parameters_and_optimizer(self):
        path, payload, run = self.checkpoint()
        env = encounter_factory(EncounterConfig('two', 2, 2, (2, 16)))()
        self.addCleanup(env.close)
        target = metadata(env, recurrent=True, actor_head='linear')
        transferred = load_stage_checkpoint(path, target, run.ppo)
        agent = RecurrentAgent(target['observation_scale'], 'linear')
        optimizer = torch.optim.Adam(agent.parameters())
        agent.load_state_dict(transferred['agent'])
        optimizer.load_state_dict(transferred['optimizer'])
        for name, value in agent.state_dict().items():
            torch.testing.assert_close(value, payload['agent'][name], rtol=0, atol=0)
        for state in optimizer.state.values():
            self.assertEqual(float(state['step']), 7)
            self.assertTrue(torch.all(state['exp_avg'] == .125))
            self.assertTrue(torch.all(state['exp_avg_sq'] == .25))
        self.assertTrue(all(torch.count_nonzero(state) == 0 for state in agent.initial_state(4)))
        for mutation in ('engine_sha256', 'observation_scale'):
            changed = deepcopy(target); changed[mutation] = 'changed'
            with self.assertRaises(ValueError):
                load_stage_checkpoint(path, changed, run.ppo)
        changed = deepcopy(target)
        changed['scenario']['reward']['kill_bonus'] += 1
        with self.assertRaises(ValueError):
            validate_transfer_metadata(payload['metadata'], changed)
        with self.assertRaises(ValueError):
            load_stage_checkpoint(path, target, replace(run.ppo, gamma=.5))

    def test_schedule_continues_without_restarting_decay(self):
        config = PPOConfig(total_timesteps=1280, anneal_lr=True)
        first = stage_schedule(config, schedule_timesteps=10000)
        payload = {'curriculum': first, 'global_step': 1280, 'ppo_config': asdict(config)}
        second = stage_schedule(config, source_payload=payload, schedule_timesteps=10000)
        self.assertEqual(second['step_offset'], 1280)
        self.assertAlmostEqual(learning_rate(config, 1, second), config.learning_rate * .872)
        resumed = stage_schedule(config, resume_payload=dict(payload, curriculum=second))
        self.assertEqual(resumed['step_offset'], 1280)
        third = stage_schedule(config, source_payload=dict(payload, curriculum=second))
        self.assertEqual(third['step_offset'], 2560)
        with self.assertRaisesRegex(ValueError, 'horizon'):
            stage_schedule(config, source_payload=payload, schedule_timesteps=20000)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            stage_schedule(replace(config, total_timesteps=10000), source_payload=payload)
        constant = replace(config, anneal_lr=False, total_timesteps=20000)
        self.assertEqual(learning_rate(constant, 1, stage_schedule(constant, source_payload=payload)),
                         constant.learning_rate)

    def test_checkpoint_reconstructs_selected_encounter_and_preflight(self):
        path, payload, run = self.checkpoint()
        env = checkpoint_factory(payload)()
        self.addCleanup(env.close)
        env.reset(seed=42)
        self.assertEqual(len(env.world.monsters), 1)
        with self.assertRaisesRegex(ValueError, 'already completed'):
            preflight(run, resume=path)
        full = load_run_config(CONFIG, 'full')
        report = preflight(full, continue_from=path)
        self.assertIsNone(report['encounter'])
        self.assertEqual(report['schedule']['step_offset'], payload['global_step'])
        with self.assertRaises(ValueError):
            preflight(full, resume=path)

    def test_cli_validation_never_calls_training_and_manual_launch_forwards_config(self):
        with patch('Training.Mystic_Sim.curriculum_run.train') as training, redirect_stdout(io.StringIO()):
            main(['--config', str(CONFIG), '--stage', 'two_innies', '--validate-only'])
            training.assert_not_called()
            # CLI wiring only: train is mocked, no rollout or optimizer executes.
            main(['--config', str(CONFIG), '--stage', 'three_innies',
                  '--device', 'cpu', '--run-dir', str(self.path / 'run')])
            args, kwargs = training.call_args
            self.assertTrue(args[0].recurrent)
            self.assertEqual(kwargs['encounter'].enemy_count, 3)
            self.assertEqual(kwargs['device'], 'cpu')
            self.assertEqual(kwargs['schedule_timesteps'], 2000000)

    def test_inherited_config_reuses_settings_without_mutating_base(self):
        base = self.write_config()
        child = self.path / 'next.json'
        child.write_text(json.dumps({'extends': base.name, 'stage': 'three_innies',
                                    'continue_from': 'previous/checkpoint_08.pt',
                                    'ppo': {'total_timesteps': 300000}}))
        run = load_run_config(child)
        self.assertEqual(run.encounter.enemy_count, 3)
        self.assertEqual(run.ppo.total_timesteps, 300000)
        self.assertEqual(run.ppo.num_envs, 4)
        self.assertEqual(run.continue_from, Path('previous/checkpoint_08.pt'))
        self.assertEqual(load_run_config(base).ppo.total_timesteps, 200000)
        child.write_text(json.dumps({'extends': child.name}))
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            load_run_config(child)

    def test_configured_continuation_fresh_override_and_stage_listing(self):
        checkpoint, _, _ = self.checkpoint()
        config = self.write_config(lambda d: d.update(stage='three_innies', continue_from=str(checkpoint)))
        with patch('Training.Mystic_Sim.curriculum_run.train') as training, redirect_stdout(io.StringIO()):
            report = main(['--config', str(config), '--validate-only'])
            self.assertEqual(report['initialization'], 'continue')
            self.assertEqual(report['schedule']['source_checkpoint'], str(checkpoint.resolve()))
            training.assert_not_called()
            main(['--config', str(config)])
            self.assertEqual(training.call_args.kwargs['continue_from'], checkpoint)
            main(['--config', str(config), '--fresh'])
            self.assertIsNone(training.call_args.kwargs['continue_from'])
            training.reset_mock()
            with patch('Training.Mystic_Sim.curriculum_run.preflight', side_effect=AssertionError('No resets needed')):
                rows = main(['--config', str(config), '--list-stages'])
            self.assertEqual([row['enemy_count'] for row in rows], [1, 2, 3, 5, 20, 40, 80])
            training.assert_not_called()


if __name__ == '__main__':
    unittest.main()
