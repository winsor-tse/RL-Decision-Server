"""Evaluate a simulator checkpoint and optionally record its Pygame gameplay."""
import argparse
from collections import deque
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .core import (Agent, accumulate, episode_metrics, finish_metrics, json_write,
                   load_checkpoint, make_env, metadata)


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
    recording_error = None
    start = time.perf_counter()
    for index, seed in enumerate(seeds):
        env, recorder = factory(), None
        try:
            obs, info = env.reset(seed=seed)
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
                    action = int(rng.integers(8))
                else:
                    with torch.no_grad():
                        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        if stochastic:
                            probabilities = torch.softmax(agent.actor(agent.normalized(tensor)), -1)[0].cpu().numpy()
                            action = int(rng.choice(8, p=probabilities.astype(float) / probabilities.sum(dtype=float)))
                        else:
                            action = int(agent.greedy(tensor).item())
                obs, reward, _, _, info = env.step(action)
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
            'damage_dealt', 'damage_taken', 'invalid_cast_rate', 'collisions', 'cooldown_rejections')
    summary = {key: float(np.mean([row[key] for row in episodes])) for key in keys}
    summary['sps_including_recording'] = sum(row['length'] for row in episodes) / max(.000001, time.perf_counter()-start)
    summary['components'] = {key: float(np.mean([row['components'][key] for row in episodes]))
                             for key in episodes[0]['components']}
    return {'policy': 'random' if agent is None else 'stochastic' if stochastic else 'greedy',
            'episodes': episodes, 'summary': summary,
            'recording': str(record_path) if record_path and not recording_error else None,
            'recording_error': recording_error}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--record', type=Path, help='Optional GIF of the first episode')
    parser.add_argument('--output', type=Path, default=Path('runs/mystic_sim/evaluation.json'))
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0: parser.error('Use positive episodes and a nonnegative seed')
    if args.device == 'cuda' and not torch.cuda.is_available(): parser.error('CUDA is unavailable')
    torch.set_num_threads(1)
    env = make_env()
    try:
        checkpoint = load_checkpoint(args.checkpoint, metadata(env))
        agent = Agent(env.observation_space.high).to(args.device)
        agent.load_state_dict(checkpoint['agent'])
    finally:
        env.close()
    agent.eval()
    seeds = list(range(args.seed, args.seed+args.episodes))
    result = evaluate(agent, seeds, device=args.device, record_path=args.record, stochastic=args.stochastic)
    result['random_baseline'] = evaluate(None, seeds)
    result['checkpoint'] = str(args.checkpoint.resolve())
    json_write(args.output, result)
    print(result['summary'])
    if result['recording_error']: print('Recording unavailable:', result['recording_error'])
    print('Evaluation:', args.output.resolve())


if __name__ == '__main__':
    main()
