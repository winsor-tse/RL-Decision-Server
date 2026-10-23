"""Start the bridge and a model-inference command."""

import argparse
from typing import Sequence

from Automation.processes import DEFAULT_CONFIG, load_config, normalize_command, run_stack


def resolve_inference_command(
    config: dict,
    override: str | Sequence[object] | None = None,
) -> str | Sequence[object]:
    """Resolve an override or an explicit algorithm-specific command."""
    if override:
        return override

    # Keep custom/smoke configurations with one explicit command supported.
    configured_command = config.get("inference_command")
    if configured_command:
        return configured_command

    algorithm = str(config.get("inference_algorithm", "")).lower()
    if not algorithm:
        raise ValueError(
            "inference_algorithm or inference_command must be configured."
        )

    algorithm_command = config.get(f"{algorithm}_inference_command")
    if not algorithm_command:
        raise ValueError(
            "No explicit inference command configured for "
            f"inference_algorithm={algorithm!r}."
        )
    return algorithm_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument('--checkpoint', help='Explicit Mystic simulator checkpoint')
    parser.add_argument('--action-mode', choices=('sampled', 'greedy'), default=None,
                        help='Checkpoint action selection (default: sampled)')
    parser.add_argument('--assume-sim-parity', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='Checkpoint device (default: cuda); use cpu or cuda:N to override')
    parser.add_argument(
        "--command",
        help="Override inference_command with a shell-style command string.",
    )
    args = parser.parse_args(argv)
    if args.checkpoint and args.command:
        parser.error('--checkpoint and --command are mutually exclusive')
    if args.assume_sim_parity and not args.checkpoint:
        parser.error('--assume-sim-parity requires --checkpoint')
    if args.action_mode is not None and not args.checkpoint:
        parser.error('--action-mode requires --checkpoint')

    config = load_config(args.config)
    if args.checkpoint:
        inference_command = ['python', '-u', '-m', 'Inference.Mystic_Live.server',
                             '--checkpoint', args.checkpoint, '--device', args.device]
        if args.assume_sim_parity:
            inference_command.append('--assume-sim-parity')
        if (args.action_mode or 'sampled') == 'sampled':
            inference_command.append('--sampled')
    else:
        inference_command = resolve_inference_command(config, args.command)
    if 'Inference.Mystic_Live.server' in normalize_command(inference_command):
        config = dict(config)
        bridge = normalize_command(config['bridge_command'])
        if 'Automation.Bridge.ws_zmq_bridge' not in bridge:
            parser.error('Mystic inference requires Automation.Bridge.ws_zmq_bridge')
        if '--single-run' not in bridge:
            bridge.append('--single-run')
        config['bridge_command'] = bridge
        config['child_ready_signal'] = 'Mystic live lifecycle listening on '
        config.setdefault('child_ready_timeout_seconds', 120)

    return run_stack(
        config,
        inference_command,
        "inference",
        start_tensorboard=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
