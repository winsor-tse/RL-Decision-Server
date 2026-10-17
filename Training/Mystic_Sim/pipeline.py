"""Run configured difficulty mixtures sequentially, transferring final checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .curriculum_run import load_run_config, preflight

ROOT = Path(__file__).resolve().parents[2]


def build_pipeline(config, *, start_stage=None, continue_from=None, resume_checkpoint=None,
                   device=None):
    path = Path(config).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or set(data) - {'version', 'stages', 'continue_from'}
            or not {'version', 'stages'}.issubset(data)
            or data['version'] != 1 or not isinstance(data['stages'], list) or not data['stages']):
        raise ValueError('Pipeline needs version 1 and a nonempty list of stage config paths')
    if continue_from and resume_checkpoint:
        raise ValueError('Choose continuation or resume, not both')
    configs = [(path.parent / p).resolve() for p in data['stages']]
    runs = [load_run_config(p) for p in configs]
    names = [r.stage for r in runs]
    if len(set(names)) != len(names):
        raise ValueError('Pipeline stage names must be unique')
    if start_stage is not None and start_stage not in names:
        raise ValueError(f'Unknown start stage: {start_stage}')
    offset = names.index(start_stage) if start_stage else 0
    configs, runs = configs[offset:], runs[offset:]
    if offset and not (continue_from or resume_checkpoint):
        raise ValueError('Starting a later stage requires --continue-from or --resume-checkpoint')
    rows = []
    configured_source = data.get('continue_from')
    if configured_source is not None and (not isinstance(configured_source, str) or not configured_source.strip()):
        raise ValueError('Pipeline continue_from must be a checkpoint path or null')
    source = (Path(continue_from).resolve() if continue_from else
              (ROOT / configured_source).resolve() if configured_source and not resume_checkpoint else None)
    resume = Path(resume_checkpoint).resolve() if resume_checkpoint else None
    directories = set()
    for index, (config_path, run) in enumerate(zip(configs, runs)):
        if run.runtime.get('stop_after_checkpoint') is not None:
            raise ValueError('Pipeline stages must finish their budgets; remove stop_after_checkpoint')
        if run.continue_from is not None:
            raise ValueError('Pipeline owns transfers; stage configs must set continue_from to null')
        directory = run.runtime.get('run_dir')
        if directory is None:
            raise ValueError('Pipeline stages need explicit, distinct runtime.run_dir values')
        directory = (ROOT / directory).resolve()
        if directory in directories:
            raise ValueError('Pipeline run directories must be distinct')
        directories.add(directory)
        stage_resume = resume if index == 0 else None
        if stage_resume and directory != stage_resume.parent.parent:
            raise ValueError('Resume checkpoint must belong to the configured stage run directory')
        if directory.exists() and any(directory.iterdir()) and not stage_resume:
            raise ValueError(f'Run directory is not empty: {directory}; use explicit resume or new directories')
        # Future transfer checkpoints do not exist yet. Validate all geometry now;
        # curriculum_run will validate each actual transfer before training starts.
        preflight(run, resume=stage_resume,
                  continue_from=source if index == 0 and not stage_resume else None)
        command = [sys.executable, '-m', 'Training.Mystic_Sim.curriculum_run',
                   '--config', str(config_path), '--run-dir', str(directory)]
        if device:
            command += ['--device', device]
        if stage_resume:
            command += ['--resume', str(stage_resume)]
        elif source:
            command += ['--continue-from', str(source)]
        else:
            command += ['--fresh']
        checkpoint = directory / 'checkpoints' / 'checkpoint_10.pt'
        rows.append({'stage': run.stage, 'command': command, 'run_dir': str(directory),
                     'checkpoint': str(checkpoint),
                     'transitions': run.ppo.updates * run.ppo.batch_size})
        source = checkpoint
    return rows


def execute_pipeline(rows):
    for row in rows:
        print(f"Starting {row['stage']}: {row['transitions']} transitions; no success gate", flush=True)
        subprocess.run(row['command'], cwd=ROOT, check=True)
        # A successful process exit alone is not sufficient if it stopped early.
        progress_path = Path(row['run_dir']) / 'progress.json'
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if (not progress.get('complete') or progress.get('global_step') != row['transitions']
                or not Path(row['checkpoint']).is_file()):
            raise RuntimeError(f"{row['stage']} did not finish its budget; refusing to advance")
        print(f"Finished {row['stage']}; transferring checkpoint 10", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--start-stage', help='Start at a named stage in the selected pipeline')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--continue-from', type=Path)
    group.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda','auto'))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    rows = build_pipeline(args.config, start_stage=args.start_stage,
                          continue_from=args.continue_from, resume_checkpoint=args.resume_checkpoint,
                          device=args.device)
    print(json.dumps(rows, indent=2))
    if not args.validate_only:
        execute_pipeline(rows)
    return rows


if __name__ == '__main__':
    main()
