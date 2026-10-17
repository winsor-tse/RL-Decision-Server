"""Fresh 1v80 LSTM PPO, no curriculum JSON: python -m Training.Mystic_Sim.train_full."""
from pathlib import Path

from .core import PPOConfig
from .train import main as train_main


def main(argv=None):
    # Resolve user-independent paths from the checkout, not a developer's home.
    # Reuse the same PPO, recurrent batching, evaluation, and checkpoint pipeline.
    defaults = PPOConfig(recurrent=True,seed=42,total_timesteps=3_000_000,
                         num_envs=4,num_steps=256,num_minibatches=4,
                         learning_rate=.00025,anneal_lr=False,
                         eval_episodes=30,eval_seed=20000)
    return train_main(argv,defaults=defaults,record_from_checkpoint=5,
                      run_root=Path(__file__).resolve().parents[2],full_only=True)


if __name__=='__main__':
    main()
