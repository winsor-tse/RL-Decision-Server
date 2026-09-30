"""Mystic-specific PPO contract, independent worlds, and checkpoint validation."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from Custom_enviornments.Mystic_Sim.actions import contract_metadata
from Custom_enviornments.Mystic_Sim.config import ScenarioConfig
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.map_loader import DEFAULT_MAP_PATH
from Custom_enviornments.Mystic_Sim.rewards import COMPONENTS


@dataclass
class PPOConfig:
    seed: int = 1
    total_timesteps: int = 262144
    num_envs: int = 1
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip_coef: float = .2
    ent_coef: float = .01
    vf_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float = .03
    anneal_lr: bool = True
    eval_episodes: int = 3
    eval_seed: int = 10000

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def updates(self):
        return math.ceil(self.total_timesteps / self.batch_size)

    def validate(self):
        for name in ('total_timesteps', 'num_envs', 'num_steps', 'num_minibatches',
                     'update_epochs', 'eval_episodes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.seed < 0 or self.eval_seed < 0:
            raise ValueError('Seeds must be nonnegative')
        if self.batch_size % self.num_minibatches or self.batch_size // self.num_minibatches < 2:
            raise ValueError('Batch size must divide evenly into minibatches of at least two samples')
        if self.updates < 10:
            raise ValueError('Use at least ten PPO updates to produce ten distinct checkpoints')
        for name in ('learning_rate', 'max_grad_norm', 'clip_coef'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('ent_coef', 'vf_coef', 'target_kl'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('gamma and gae_lambda must be in [0, 1]')


def checkpoint_updates(updates):
    """Exactly ten distinct, approximately evenly spaced safe update boundaries."""
    if updates < 10:
        raise ValueError('Ten checkpoints require at least ten updates')
    return {math.ceil(i * updates / 10): i for i in range(1, 11)}


def make_env():
    # No free-play overrides, reward normalization, live imports, or action masks.
    return MysticSimEnv(config=ScenarioConfig(), trace=False)


class SimBatch:
    """Simple synchronous batching; each world owns its RNG, scheduler, and state.

    Explicit resets avoid heterogeneous Gym info collation and autoreset ambiguity.
    Step observations are always the final pre-reset observations. Phase 7 may
    replace this adapter with subprocess workers without changing PPO targets.
    """
    def __init__(self, count, seed, factory=make_env):
        self.envs = []
        try:
            for _ in range(count):
                self.envs.append(factory())
            self.observations = np.stack([e.reset(seed=seed+i)[0] for i, e in enumerate(self.envs)])
        except Exception:
            self.close()
            raise

    def step(self, actions):
        if len(actions) != len(self.envs):
            raise ValueError('One action per environment is required')
        rows = [e.step(int(a)) for e, a in zip(self.envs, actions)]
        final_obs = np.stack([row[0] for row in rows])
        rewards = np.asarray([row[1] for row in rows], dtype=np.float32)
        terminated = np.asarray([row[2] for row in rows], dtype=bool)
        truncated = np.asarray([row[3] for row in rows], dtype=bool)
        infos = [row[4] for row in rows]
        self.observations = final_obs.copy()
        for i in np.flatnonzero(terminated | truncated):
            self.observations[i] = self.envs[i].reset()[0]
        return final_obs, rewards, terminated, truncated, infos

    def close(self):
        for env in self.envs:
            env.close()


def layer(inputs, outputs, gain=np.sqrt(2)):
    linear = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)
    return linear


class Agent(nn.Module):
    def __init__(self, observation_high):
        super().__init__()
        self.register_buffer('observation_scale', torch.tensor(observation_high, dtype=torch.float32))
        self.actor = nn.Sequential(layer(26, 128), nn.Tanh(), layer(128, 128), nn.Tanh(), layer(128, 8, .01))
        self.critic = nn.Sequential(layer(26, 128), nn.Tanh(), layer(128, 128), nn.Tanh(), layer(128, 1, 1))

    def normalized(self, obs):
        # Fixed per-feature scaling, checkpointed with the model; no running stats.
        return obs / self.observation_scale

    def value(self, obs):
        return self.critic(self.normalized(obs)).squeeze(-1)

    def action_value(self, obs, action=None):
        x = self.normalized(obs)
        distribution = Categorical(logits=self.actor(x))
        action = distribution.sample() if action is None else action
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(x).squeeze(-1)

    def greedy(self, obs):
        return self.actor(self.normalized(obs)).argmax(dim=-1)


def advantages(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Bootstrap truncations from final states, but never propagate GAE across reset."""
    result = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        tail = delta + gamma * gae_lambda * (~(terminated[t] | truncated[t])) * tail
        result[t] = tail
    return result, result + values


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def metadata(env):
    import inspect
    source = Path(inspect.getfile(MysticSimEnv)).parent
    return {
        'backend': 'mystic_sim_python', 'architecture': 'ppo_mlp_128x128_v1',
        'contract': contract_metadata(),
        'observation_dtype': 'float32', 'preprocessing': 'divide_by_box_high_v1',
        'observation_scale': env.observation_space.high.tolist(),
        'scenario': json.loads(json.dumps(asdict(env.config))),
        'map_sha256': hashlib.sha256(Path(DEFAULT_MAP_PATH).read_bytes()).hexdigest(),
        'mechanics_sha256': hashlib.sha256((source / 'mechanics_manifest.yaml').read_bytes()).hexdigest(),
        'engine_sha256': hashlib.sha256(b''.join(
            p.name.encode() + p.read_bytes() for p in sorted(source.glob('*.py'))
            if p.name != 'viewer.py')).hexdigest(),
        'resume_policy': 'optimizer/counters/RNG restored; fresh simulator episodes, not exact world replay',
    }


def load_checkpoint(path, expected_metadata):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != 'mystic_sim_ppo_v1':
        raise ValueError('Expected a Mystic Sim PPO checkpoint; live/legacy weights are incompatible')
    if payload.get('metadata') != expected_metadata:
        raise ValueError('Checkpoint simulator configuration, map, mechanics, or policy contract is incompatible')
    required = {'agent', 'optimizer', 'ppo_config', 'update', 'global_step', 'rng', 'episodes', 'wins'}
    if not required <= payload.keys():
        raise ValueError('Incomplete Mystic Sim training checkpoint')
    return payload


def episode_metrics():
    return dict(episode_return=0., length=0, damage_dealt=0., damage_taken=0.,
                cast_attempts=0, invalid_casts=0, collisions=0, cooldown_rejections=0,
                components={name: 0. for name in COMPONENTS})


def accumulate(metrics, action, reward, info):
    metrics['episode_return'] += float(reward)
    metrics['length'] += 1
    metrics['cast_attempts'] += int(action >= 5)
    metrics['invalid_casts'] += int(action >= 5 and not info['action_applied'])
    metrics['collisions'] += int(info['collision_kind'] is not None)
    metrics['cooldown_rejections'] += int(info['cooldown_penalty'] < 0)
    for event in info['damage_events']:
        if event['attacker_id'] == 1 and event['target_id'] != 1:
            metrics['damage_dealt'] += event['damage']
        if event['target_id'] == 1:
            metrics['damage_taken'] += event['damage']
    for key in COMPONENTS:
        metrics['components'][key] += info['reward_components'][key]


def finish_metrics(metrics, info):
    return {**metrics, 'kills': info['kills'], 'outcome': info['episode_outcome'],
            'end_reason': info['episode_end_reason'], 'survival_seconds': info['simulation_time_ms'] / 1000,
            'win': float(info['episode_outcome'] == 'win'),
            'survived': float(info['episode_outcome'] != 'loss'),
            'invalid_cast_rate': metrics['invalid_casts'] / max(1, metrics['cast_attempts'])}
