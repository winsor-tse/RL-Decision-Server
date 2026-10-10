"""PPO recurrent sequences, adapted from PPO_lstm_server.py for N worlds.

Shuffle contiguous TIME BLOCKS only, keeping every environment's chronology.
Each block receives its saved pre-observation hidden/cell state. Episode-start
masks reset individual worlds inside blocks; rollout boundaries do not reset.
Actor and critic have disjoint recurrent trunks for independent gradient clips.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .core import layer
from Custom_enviornments.Mystic_Sim.actions import ACTIONS


class RecurrentBranch(nn.Module):
    def __init__(self, size, outputs):
        super().__init__()
        width = size * 8
        self.network = nn.Sequential(layer(size, width), nn.ReLU(),
                                     layer(width, width), nn.ReLU(), layer(width, 512))
        self.lstm = nn.LSTM(512, 128)
        for name, parameter in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(parameter)
            else:
                nn.init.orthogonal_(parameter)
        self.head = nn.Sequential(layer(128, 64), nn.Tanh(), layer(64, 64), nn.Tanh(),
                                  layer(64, outputs, .01 if outputs == len(ACTIONS) else 1.))

    def forward(self, observations, state, starts):
        embedded = self.network(observations)
        outputs = []
        for embedding, start in zip(embedded, starts):
            mask = (~start.bool()).to(embedding.dtype).view(1, -1, 1)
            output, state = self.lstm(embedding.unsqueeze(0), tuple(x * mask for x in state))
            outputs.append(output)
        hidden = self.head[:4](torch.cat(outputs))
        return self.head[4](hidden), state, hidden


class RecurrentAgent(nn.Module):
    recurrent = True

    def __init__(self, observation_high):
        super().__init__()
        self.register_buffer('observation_scale', torch.tensor(observation_high, dtype=torch.float32))
        self.actor = RecurrentBranch(len(observation_high), len(ACTIONS))
        self.critic = RecurrentBranch(len(observation_high), 1)

    def initial_state(self, count):
        return tuple(torch.zeros(1, count, 128, device=self.observation_scale.device) for _ in range(4))

    def sequence(self, obs, state, starts):
        x = obs / self.observation_scale
        logits, actor_state, _ = self.actor(x, state[:2], starts)
        value, critic_state, hidden = self.critic(x, state[2:], starts)
        return logits, value.squeeze(-1), (*actor_state, *critic_state), hidden

    def step(self, obs, state, starts, action=None):
        logits, value, state, _ = self.sequence(obs.unsqueeze(0), state, starts.unsqueeze(0))
        distribution = Categorical(logits=logits[0])
        action = distribution.sample() if action is None else action
        return action, distribution.log_prob(action), distribution.entropy(), value[0], state


def recurrent_minibatches(num_steps, num_minibatches):
    if num_minibatches < 1 or num_steps % num_minibatches:
        raise ValueError('num_steps must be divisible by num_minibatches')
    length = num_steps // num_minibatches
    starts = np.arange(0, num_steps, length)
    np.random.shuffle(starts)
    return [slice(int(start), int(start + length)) for start in starts]


def optimize_recurrent(agent, optimizer, config, observations, actions, old_logprobs,
                       values, adv, returns, states, starts):
    from .train import critic_loss, clip_gradients
    rows = []
    for _ in range(config.update_epochs):
        epoch_kl = []
        for ix in recurrent_minibatches(config.num_steps, config.num_minibatches):
            # [time, env, feature], never shuffled individual transitions.
            initial = tuple(s[ix.start].detach() for s in states)
            logits, value, _, _ = agent.sequence(observations[ix], initial, starts[ix])
            distribution = Categorical(logits=logits)
            logratio = distribution.log_prob(actions[ix]) - old_logprobs[ix]
            ratio = logratio.exp()
            advantage = adv[ix]
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            policy_loss = torch.maximum(-advantage * ratio,
                -advantage * ratio.clamp(1-config.clip_coef, 1+config.clip_coef)).mean()
            value_loss = critic_loss(value, values[ix], returns[ix], config.value_clip_coef)
            entropy = distribution.entropy().mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite recurrent PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_grad, critic_grad = clip_gradients(agent, config)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio-1)-logratio).mean()
                clipfrac = ((ratio-1).abs() > config.clip_coef).float().mean()
                rows.append([float(x) for x in (policy_loss, value_loss, entropy, kl,
                                                clipfrac, actor_grad, critic_grad)])
                epoch_kl.append(float(kl))
        if config.target_kl and np.mean(epoch_kl) > config.target_kl:
            break
    result = dict(zip(('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clipfrac',
                       'actor_grad_norm', 'critic_grad_norm'), np.mean(rows, axis=0).tolist()))
    variance = returns.var(unbiased=False)
    result['explained_variance'] = float(1-(returns-values).var(unbiased=False)/variance) if variance > 0 else 0.
    with torch.no_grad():
        _, predictions, _, hidden = agent.sequence(observations, tuple(s[0] for s in states), starts)
        result['critic_value_std'] = float(predictions.std(unbiased=False))
        result['critic_target_std'] = float(returns.std(unbiased=False))
        result['critic_hidden_saturation'] = float((hidden.abs() > .99).float().mean())
        result['post_update_explained_variance'] = float(1-(returns-predictions).var(unbiased=False)/variance) if variance > 0 else 0.
    return result
