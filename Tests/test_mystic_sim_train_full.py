"""Portable full-scenario launcher tests; never run PPO optimization."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Training.Mystic_Sim.train_full import main
from Training.Mystic_Sim.core import PPOConfig


class FullTrainingTests(unittest.TestCase):
    def test_default_preflight_is_full_fresh_without_training(self):
        with patch('Training.Mystic_Sim.train.train') as train, redirect_stdout(io.StringIO()):
            report=main(['--validate-only','--device','cpu'])
        train.assert_not_called()
        self.assertEqual((report['enemy_count'],report['kill_goal'],report['max_episode_steps']),(80,5,1024))
        self.assertTrue(report['ppo_config']['recurrent'])
        self.assertFalse(report['ppo_config']['anneal_lr'])
        self.assertEqual(report['actual_timesteps'],3000320)
        self.assertEqual(report['record_from_checkpoint'],5)
        self.assertFalse(Path(report['run_dir']).exists())

    def test_overrides_and_paths_independent_of_cwd(self):
        from Training.Mystic_Sim import train_full
        root=Path(train_full.__file__).resolve().parents[2]
        old=os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                with patch('Training.Mystic_Sim.train.train') as train, redirect_stdout(io.StringIO()):
                    main(['--device','cpu','--num-envs','2','--seed','7',
                          '--total-timesteps','10000','--run-dir','runs/full_test',
                          '--no-record-gameplay','--eval-episodes','2'])
                config=train.call_args.args[0]
                self.assertEqual((config.num_envs,config.seed,config.total_timesteps),(2,7,10000))
                self.assertEqual(train.call_args.kwargs['run_dir'],root/'runs/full_test')
                self.assertFalse(train.call_args.kwargs['record_gameplay'])
                self.assertIsNone(train.call_args.kwargs['resume'])
            finally:
                os.chdir(old)

    def test_curriculum_resume_rejected(self):
        raw={'ppo_config':asdict(PPOConfig()),'metadata':{'scenario':{'encounter':{
            'stage_id':'easy','enemy_count':1,'kill_goal':1,'path_distance':[2,6]}}}}
        with patch('Training.Mystic_Sim.train.torch.load',return_value=raw), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--resume','checkpoint.pt','--validate-only','--device','cpu'])

    def test_resume_uses_saved_hyperparameters(self):
        config=PPOConfig(recurrent=True,seed=8,total_timesteps=20000,num_envs=2)
        raw={'ppo_config':asdict(config),'metadata':{'scenario':{'encounter':None}},'update':2}
        with patch('Training.Mystic_Sim.train.torch.load',return_value=raw), patch(
                'Training.Mystic_Sim.train.load_checkpoint',return_value=raw), redirect_stdout(io.StringIO()):
            report=main(['--resume','checkpoint.pt','--validate-only','--device','cpu'])
        self.assertEqual(report['ppo_config'],asdict(config))
        self.assertEqual(report['initialization'],'resume')

    def test_full_world_gif_dependency_smoke(self):
        from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
        from Custom_enviornments.Mystic_Sim.config import ScenarioConfig
        from Training.Mystic_Sim.evaluate import evaluate
        base=ScenarioConfig()
        def factory():
            return MysticSimEnv(config=replace(base,reward=replace(base.reward,max_episode_steps=2)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'smoke.gif'
            report=evaluate(None,[42],factory=factory,record_path=path)
            self.assertIsNone(report['recording_error'])
            self.assertTrue(path.is_file())


if __name__=='__main__':
    unittest.main()
