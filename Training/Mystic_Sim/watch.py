"""Watch a frozen checkpoint in the existing Pygame viewer; never train."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from Custom_enviornments.Mystic_Sim.observation import encode_observation
from Custom_enviornments.Mystic_Sim.viewer import PlaySession
from .evaluate import load_evaluation_agent
from .history import history_class
from .watch_state import PolicyDecision, snapshot_from_session
from .watch_dashboard import PolicyViewer


class PolicySession(PlaySession):
    """Same episode rules, inference history and sampling RNG as evaluate()."""
    def __init__(self, agent, factory, *, seed=20000, mode='sampled', device='cpu'):
        self.agent, self.mode, self.device = agent, mode, device
        self.env = factory()
        self.seed, self.training_rules, self.speed = seed, True, 1.0
        self.recurrent = bool(getattr(agent, 'recurrent', False))
        self.reset()

    def reset(self, new_seed=False):
        super().reset(new_seed)
        self.history = history_class(getattr(self.agent, 'history_mode', 'confirmed')).from_env(self.env) if self.recurrent else None
        self.state = self.agent.initial_state(1) if self.recurrent else None
        self.episode_start = True
        self.rng = np.random.default_rng(self.seed + 1000000)
        self.last_action = None
        self.log.append(f'Policy: {self.mode}, seed {self.seed}. Manual actions disabled.')
        self.decision = None
        self.snapshot = snapshot_from_session(self)

    def step(self, action=None):
        # Keyboard movement/spells never override the frozen policy.
        if self.paused or self.env.episode_done:
            return None
        obs = encode_observation(self.env.world)
        policy_obs = self.history.encode(obs) if self.recurrent else obs
        with torch.no_grad():
            tensor = torch.as_tensor(policy_obs, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            if self.recurrent:
                logits, _, self.state, _ = self.agent.sequence(
                    tensor.unsqueeze(0), self.state,
                    torch.tensor([[self.episode_start]], dtype=torch.bool, device=self.device))
                self.episode_start = False
                probabilities = torch.softmax(logits[0, 0], -1).cpu().numpy()
            else:
                probabilities = torch.softmax(self.agent.actor(
                    self.agent.normalized(tensor)), -1)[0].cpu().numpy()
            action = (int(self.rng.choice(len(probabilities), p=probabilities.astype(float)
                          / probabilities.sum(dtype=float))) if self.mode == 'sampled'
                      else int(probabilities.argmax()))
        # Retain the exact distribution and input that produced this action.
        # The dashboard reads these copies; drawing never advances memory/RNG.
        self.decision = PolicyDecision(self.snapshot, tuple(float(p) for p in probabilities), action)
        self.last_action = action
        info = super().step(action)
        if self.history is not None:
            self.history.update_from_info(obs, action, info)
        self.snapshot = snapshot_from_session(self)
        if self.env.episode_done:
            print(f'Seed {self.seed} / {self.mode}: {info["episode_outcome"]}, '
                  f'reason={info["episode_end_reason"]}, '
                  f'steps={self.env.world.step_count}, return={self.total_reward:.2f}', flush=True)
        return info


def parse_watch_args(argv=None, *, root=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Example: Watch_Mystic_Policy.bat --checkpoint '
        'runs/my_run/checkpoints/checkpoint_09.pt --seed 2000 --mode sampled'))
    parser.add_argument('checkpoint_path', nargs='*', metavar='PATH',
                        help='Checkpoint file (or use --checkpoint PATH)')
    parser.add_argument('--checkpoint', type=Path, help='Checkpoint file to watch')
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--mode', choices=('sampled', 'greedy'), default='sampled')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--smoke-test', action='store_true', help='Render 20 decisions with SDL dummy driver')
    parser.add_argument('--screenshot', type=Path, help='Save the final dashboard frame as PNG')
    args = parser.parse_intermixed_args(argv)
    paths = args.checkpoint_path
    # Also accept the natural spelling "checkpoint PATH" used with the BAT.
    if len(paths) == 2 and paths[0] == 'checkpoint':
        paths = paths[1:]
    if len(paths) > 1 or (paths and args.checkpoint is not None):
        parser.error('Provide one checkpoint: --checkpoint PATH or a positional PATH')
    if args.checkpoint is None:
        if not paths:
            parser.error('A checkpoint is required; use --checkpoint PATH')
        args.checkpoint = Path(paths[0])
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    checkpoint = (root / args.checkpoint).resolve()
    if not checkpoint.is_file():
        # A common typo leaves out the run's checkpoints/ directory. Resolve
        # only that exact filename, never select a different/latest model.
        nested = checkpoint.parent / 'checkpoints' / checkpoint.name
        if nested.is_file():
            print(f'Using checkpoint from checkpoints subfolder: {nested}', flush=True)
            checkpoint = nested
        else:
            parser.error(f'Checkpoint file not found: {checkpoint}. '
                         'Saved models are normally in RUN/checkpoints/checkpoint_XX.pt')
    args.checkpoint = checkpoint
    return args


def main(argv=None):
    args = parse_watch_args(argv)
    checkpoint = args.checkpoint
    torch.set_num_threads(1)
    agent, factory, _ = load_evaluation_agent(checkpoint, args.device)
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    if args.smoke_test:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    import pygame
    session = PolicySession(agent, factory, seed=args.seed, mode=args.mode, device=args.device)
    try:
        viewer = PolicyViewer(pygame, session)
        viewer.checkpoint_name = checkpoint.name
        # Disable gameplay keys while retaining pause, speed, camera and reset controls.
        viewer.keys.clear()
        print(f'Checkpoint: {checkpoint}\nMode: {args.mode}; seed: {args.seed}\n'
              'Space: play/pause; N: single step while paused; R: replay seed; Shift+R: next seed; +/-: speed; '
              'Tab: overview; B: spawn areas; C: range; Esc: quit.', flush=True)
        if args.smoke_test:
            session.paused = False
            for _ in range(20):
                info = session.step()
                if info:
                    viewer.capture_effects(info)
                viewer.draw(.2)
            print('Policy viewer smoke test passed.')
        else:
            clock = pygame.time.Clock()
            while viewer.running:
                elapsed = clock.tick(60) / 1000
                viewer.events()
                # Avoid manual-input lookup in Viewer.advance; policy drives each decision.
                if session.paused or session.env.episode_done:
                    viewer.accumulator = 0
                else:
                    viewer.accumulator += min(elapsed, .25) * session.speed
                    while viewer.accumulator >= .2 and not session.env.episode_done:
                        info = session.step()
                        viewer.accumulator -= .2
                        if info:
                            viewer.capture_effects(info)
                pygame.display.set_caption(f'Mystic policy | {checkpoint.name} | '
                                           f'{args.mode} | seed {session.seed}')
                viewer.draw(elapsed)
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(viewer.screen, str(args.screenshot))
    finally:
        session.close()
        pygame.quit()


if __name__ == '__main__':
    main()
