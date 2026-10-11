"""Evaluate a simulator checkpoint and optionally record its Pygame gameplay."""
import argparse
from collections import deque, Counter
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .core import (Agent, accumulate, episode_metrics, finish_metrics, json_write, saved_config,
                   load_checkpoint, make_env, metadata)
from Custom_enviornments.Mystic_Sim.actions import ACTIONS
from .history import ObservableHistory
from .recurrent import RecurrentAgent
from .encounter_diagnostics import EncounterDiagnostics, METRICS as ENCOUNTER_METRICS


class GameplayRecorder:
    """Hidden Pygame rendering to a bounded-size GIF; no FFmpeg dependency."""
    def __init__(self, env, seed, path, stride=5):
        self.path, self.stride = Path(path), stride
        self.frames, self.steps = [], []
        self.old_driver = os.environ.get('SDL_VIDEODRIVER')
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        try:
            import pygame
            from PIL import Image
            from Custom_enviornments.Mystic_Sim.viewer import Viewer
            self.pg, self.Image = pygame, Image
            self.session = SimpleNamespace(env=env, seed=seed, training_rules=True,
                                           paused=False, speed=1., log=deque(maxlen=5),
                                           info={}, total_reward=0.)
            self.viewer = Viewer(pygame, self.session)
        except Exception:
            self.close()
            raise

    def capture(self, info, total_reward, force=False):
        step = self.session.env.world.step_count
        self.session.info, self.session.total_reward = info, total_reward
        if step:
            self.viewer.capture_effects(info)
        if not force and step % self.stride:
            # Avoid retaining transient animation data across skipped frames.
            self.viewer.flashes.clear()
            self.viewer.floating.clear()
            return
        self.pg.event.pump()
        self.viewer.draw(.2 * self.stride)
        surface = self.pg.transform.smoothscale(self.viewer.screen, (660, 430))
        frame = self.Image.frombytes('RGB', surface.get_size(), self.pg.image.tobytes(surface, 'RGB'))
        self.frames.append(frame.quantize(colors=128))
        self.steps.append(step)

    def save(self):
        if not self.frames:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        durations = [max(20, (b-a)*200) for a, b in zip(self.steps, self.steps[1:])] + [1000]
        temporary = self.path.with_suffix('.tmp')
        self.frames[0].save(temporary, format='GIF', save_all=True, append_images=self.frames[1:],
                            duration=durations, loop=0, optimize=False, disposal=2)
        temporary.replace(self.path)

    def close(self):
        if hasattr(self, 'pg'):
            self.pg.quit()
        if self.old_driver is None:
            os.environ.pop('SDL_VIDEODRIVER', None)
        else:
            os.environ['SDL_VIDEODRIVER'] = self.old_driver
        self.frames.clear()


def evaluate(agent, seeds, *, device='cpu', record_path=None, record_stride=5,
             stochastic=False, factory=make_env):
    """Use separate worlds and local action RNG: evaluation never alters training RNG."""
    if not seeds or record_stride < 1:
        raise ValueError('Evaluation needs at least one seed and a positive recording stride')
    episodes = []
    action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
    failures = Counter()
    probability_rows = []
    successful_moves = 0
    cooldown_probability_sum = Counter()
    cooldown_observation_count = Counter()
    recording_error = None
    start = time.perf_counter()
    for index, seed in enumerate(seeds):
        env, recorder = factory(), None
        try:
            obs, info = env.reset(seed=seed)
            encounter_diagnostics = EncounterDiagnostics(env.world)
            recurrent = bool(getattr(agent, 'recurrent', False))
            history = ObservableHistory.from_env(env) if recurrent else None
            state = agent.initial_state(1) if recurrent else None
            episode_start = True
            rng = np.random.default_rng(seed + 1000000)
            if record_path and index == 0:
                try:
                    recorder = GameplayRecorder(env, seed, record_path, record_stride)
                    recorder.capture(info, 0, force=True)
                except Exception as exc:
                    recording_error = f'{type(exc).__name__}: {exc}'
                    if recorder: recorder.close()
                    recorder = None
            metrics = episode_metrics()
            while not env.episode_done:
                if agent is None:
                    action = int(rng.integers(len(ACTIONS)))
                    probabilities = np.full(len(ACTIONS), 1 / len(ACTIONS))
                else:
                    with torch.no_grad():
                        policy_obs = history.encode(obs) if recurrent else obs
                        tensor = torch.as_tensor(policy_obs, dtype=torch.float32, device=device).unsqueeze(0)
                        if recurrent:
                            logits, _, state, _ = agent.sequence(tensor.unsqueeze(0), state,
                                torch.tensor([[episode_start]], dtype=torch.bool, device=device))
                            episode_start = False
                            probabilities = torch.softmax(logits[0,0], -1).cpu().numpy()
                            action = int(rng.choice(len(ACTIONS), p=probabilities.astype(float) / probabilities.sum(dtype=float))) if stochastic else int(logits[0,0].argmax())
                        elif stochastic:
                            probabilities = torch.softmax(agent.actor(agent.normalized(tensor)), -1)[0].cpu().numpy()
                            action = int(rng.choice(len(ACTIONS), p=probabilities.astype(float) / probabilities.sum(dtype=float)))
                        else:
                            probabilities = torch.softmax(agent.actor(agent.normalized(tensor)), -1)[0].cpu().numpy()
                            action = int(probabilities.argmax())
                probability_rows.append(probabilities)
                # Diagnostic-only oracle. Never appended to policy observations.
                player, now = env.world.player, env.world.time_ms
                for spell in env.config.spells:
                    ready = (now >= player.cooldowns.global_ready_at_ms
                             and now >= player.cooldowns.slots.get(spell.slot, 0)
                             and (not spell.family or now >= player.cooldowns.families.get(spell.family, 0)))
                    key = f'castSpell_{spell.slot}/cooldown_{"ready" if ready else "blocked"}'
                    cooldown_probability_sum[key] += float(probabilities[spell.slot+3])
                    cooldown_observation_count[key] += 1
                action_counts[action] += 1
                previous_obs = obs
                encounter_diagnostics.before(env.world)
                obs, reward, _, _, info = env.step(action)
                info['combat_navigation'] = encounter_diagnostics.after(env.world, info)
                successful_moves += int(action < 4 and info['action_applied'])
                if not info['action_applied']:
                    failures[f"{ACTIONS[action].replace(':', '_')}/{info['action_failure_reason'] or 'unknown'}"] += 1
                if recurrent:
                    history.update_from_info(previous_obs, action, info, env.config.timing.step_ms)
                accumulate(metrics, action, reward, info)
                if recorder:
                    try:
                        recorder.capture(info, metrics['episode_return'], force=env.episode_done)
                    except Exception as exc:
                        recording_error = f'{type(exc).__name__}: {exc}'
                        recorder.close()
                        recorder = None
            episodes.append({'seed': seed, **finish_metrics(metrics, info)})
            if recorder:
                try:
                    recorder.save()
                except Exception as exc:
                    recording_error = f'{type(exc).__name__}: {exc}'
        finally:
            if recorder: recorder.close()
            env.close()
    keys = ('episode_return', 'length', 'kills', 'win', 'survived', 'survival_seconds',
            'damage_dealt', 'damage_taken', 'invalid_cast_rate', 'collisions', 'cooldown_rejections') + ENCOUNTER_METRICS
    summary = {key: float(np.mean([row[key] for row in episodes])) for key in keys}
    summary['sps_including_recording'] = sum(row['length'] for row in episodes) / max(.000001, time.perf_counter()-start)
    summary['components'] = {key: float(np.mean([row['components'][key] for row in episodes]))
                             for key in episodes[0]['components']}
    probabilities = np.asarray(probability_rows, dtype=np.float64)
    diagnostics = {'successful_moves': successful_moves,
                   'movement_attempt_rate': float(action_counts[:4].sum()/action_counts.sum()),
                   'probability_std_mean': float(probabilities.std(0).mean())}
    for i, label in enumerate(ACTIONS):
        label = label.replace(':', '_')
        diagnostics[f'action_fraction/{label}'] = float(action_counts[i]/action_counts.sum())
        diagnostics[f'probability_mean/{label}'] = float(probabilities[:,i].mean())
        diagnostics[f'probability_std/{label}'] = float(probabilities[:,i].std())
    diagnostics.update({f'failure_count/{key}': count for key,count in failures.items()})
    for key, count in cooldown_observation_count.items():
        diagnostics[f'conditional_probability/{key}'] = cooldown_probability_sum[key]/count
        diagnostics[f'conditional_observations/{key}'] = count
    return {'policy': 'random' if agent is None else 'stochastic' if stochastic else 'greedy',
            'diagnostics': diagnostics,
            'episodes': episodes, 'summary': summary,
            'recording': str(record_path) if record_path and not recording_error else None,
            'recording_error': recording_error}


def load_evaluation_agent(path, device='cpu'):
    """Load frozen weights after checking the checkpoint's original env contract."""
    from .curriculum import checkpoint_factory
    raw = torch.load(path, map_location='cpu', weights_only=True)
    factory = checkpoint_factory(raw)
    env = factory()
    try:
        config = saved_config(raw)
        contract = metadata(env, recurrent=config.recurrent, actor_head=config.actor_head)
        checkpoint = load_checkpoint(path, contract)
        agent = (RecurrentAgent(contract['observation_scale'], config.actor_head)
                 if config.recurrent else Agent(contract['observation_scale'])).to(device)
        agent.load_state_dict(checkpoint['agent'])
    finally:
        env.close()
    agent.eval()
    return agent, factory, contract


def evaluation_line(label, report):
    summary = report['summary']
    return (f'{label}: return={summary["episode_return"]:.2f}, '
            f'win={summary["win"]:.1%}, kills={summary["kills"]:.2f}, '
            f'steps={summary["length"]:.1f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--full-scenario', action='store_true',
                        help='Evaluate on the original 80-Innie task instead of the checkpoint encounter')
    parser.add_argument('--record', type=Path, help='Optional GIF of the first episode')
    parser.add_argument('--output', type=Path, default=Path('runs/mystic_sim/evaluation.json'))
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0: parser.error('Use positive episodes and a nonnegative seed')
    if args.device == 'cuda' and not torch.cuda.is_available(): parser.error('CUDA is unavailable')
    torch.set_num_threads(1)
    agent, factory, contract = load_evaluation_agent(args.checkpoint, args.device)
    seeds = list(range(args.seed, args.seed+args.episodes))
    selected_factory = make_env if args.full_scenario else factory
    result = evaluate(agent, seeds, device=args.device, record_path=args.record,
                      stochastic=args.stochastic, factory=selected_factory)
    result['random_baseline'] = evaluate(None, seeds, factory=selected_factory)
    result['encounter'] = None if args.full_scenario else contract['scenario'].get('encounter')
    result['checkpoint'] = str(args.checkpoint.resolve())
    json_write(args.output, result)
    print(result['summary'])
    if result['recording_error']: print('Recording unavailable:', result['recording_error'])
    print('Evaluation:', args.output.resolve())


if __name__ == '__main__':
    main()
