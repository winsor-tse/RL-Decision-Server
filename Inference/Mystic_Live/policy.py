"""Frozen simulator actors for the live controller; no optimizer or world stepping."""
from pathlib import Path
import hashlib

import numpy as np
import torch

from Custom_enviornments.Mystic_Sim.actions import ACTIONS, ACTION_SCHEMA
from Custom_enviornments.Mystic_Sim.map_loader import DEFAULT_MAP_PATH
from Training.Mystic_Sim.evaluate import load_evaluation_agent
from .inputs import LivePolicyInputs, PREPROCESSING
from .lifecycle import PolicyChoice


class FrozenPolicy:
    def __init__(self, checkpoint, *, device='cpu', sampled=False):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.agent, _, self.metadata = load_evaluation_agent(self.checkpoint, device)
        self.recurrent = bool(getattr(self.agent, 'recurrent', False))
        self.input_size = self.metadata['contract']['observation_size']
        self.input_adapter = LivePolicyInputs(self.input_size)
        self.sampled = sampled
        self._validate_live_contract()
        self.agent.eval().requires_grad_(False)

    def _validate_live_contract(self):
        adapter, meta = self.input_adapter, self.metadata
        expected = dict(action_schema=ACTION_SCHEMA, actions=list(ACTIONS),
                        observation_schema=adapter.schema, observation_size=self.input_size)
        if meta.get('contract') != expected:
            raise ValueError('Checkpoint action/observation contract does not match live inputs')
        if meta.get('preprocessing') != PREPROCESSING or meta.get('observation_dtype') != 'float32':
            raise ValueError('Unsupported live preprocessing or dtype')
        if not np.array_equal(np.asarray(meta['observation_scale']), adapter.high):
            raise ValueError('Checkpoint normalization differs from live input bounds')
        scale = self.agent.observation_scale.detach().cpu().numpy()
        if not np.array_equal(scale, adapter.high):
            raise ValueError('Checkpoint weight normalization buffer differs from metadata')
        if meta.get('map_sha256') != hashlib.sha256(Path(DEFAULT_MAP_PATH).read_bytes()).hexdigest():
            raise ValueError('Checkpoint map does not match the live map')
        scenario = meta.get('scenario', {})
        if (scenario.get('map_id'), scenario.get('width'), scenario.get('height')) != (53, 100, 100):
            raise ValueError('Unsupported live map configuration')
        if self.recurrent and scenario.get('terrain_collision') is not True:
            raise ValueError('Checkpoint terrain sensors do not match live terrain')
        if any(not torch.isfinite(tensor).all().item() for tensor in self.agent.state_dict().values()):
            raise ValueError('Checkpoint contains nonfinite weights')

    def initial_state(self):
        if not self.recurrent:
            return None
        lstm = self.agent.actor.lstm
        return tuple(torch.zeros(lstm.num_layers, 1, lstm.hidden_size,
                                 device=self.agent.observation_scale.device) for _ in range(2))

    @torch.inference_mode()
    def choose(self, observation, history, memory):
        # History is already encoded by LivePolicyInputs; do not append it again.
        raw = np.asarray(observation, dtype=np.float32)
        if raw.shape != (self.input_size,) or not np.isfinite(raw).all():
            raise ValueError('Invalid policy observation shape or values')
        if np.any(raw < self.input_adapter.low) or np.any(raw > self.input_adapter.high):
            raise ValueError('Policy observation outside live bounds')
        tensor = torch.as_tensor(raw, device=self.agent.observation_scale.device).unsqueeze(0)
        normalized = tensor / self.agent.observation_scale
        if self.recurrent:
            if memory is None:
                raise ValueError('Recurrent actor memory must be initialized once per run')
            starts = torch.zeros((1, 1), dtype=torch.bool, device=tensor.device)
            logits, next_memory, _ = self.agent.actor(normalized.unsqueeze(0), memory, starts)
            logits = logits[0, 0]
        else:
            logits, next_memory = self.agent.actor(normalized)[0], None
        if logits.shape != (len(ACTIONS),) or not torch.isfinite(logits).all():
            raise ValueError('Invalid actor logits')
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if self.sampled else logits.argmax()
        return PolicyChoice(int(action.item()), next_memory,
                            tuple(distribution.probs.cpu().tolist()))
