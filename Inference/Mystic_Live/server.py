"""Single-run REP server with observation-only or frozen-checkpoint inference.

56/71-input checkpoints use snapshots; 80-input checkpoints select confirmed or simulated history.
Run the matching bridge with --single-run for connection identity/events.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import threading
import time

import zmq

from .lifecycle import SingleRun
from .inputs import LivePolicyInputs


def decode_message(raw):
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON number: {value}')
    return json.loads(raw, parse_constant=reject_constant)


def serve(run, socket, *, stop_event=None, drain_seconds=1., poll_ms=50):
    """REP ordering is recv -> exactly one send, including terminal requests.

    After ending, briefly drain already queued exchanges with inert replies.
    The single-run bridge latches terminal status for all later browser ticks.
    """
    stop_event = stop_event or threading.Event()
    ended_at = None
    received = 0
    last_wait_reason = None
    announced_running = False
    while True:
        if stop_event.is_set():
            run.stop('manual_stop')
        run.check_deadlines()
        if run.terminal and ended_at is None:
            ended_at = time.monotonic()
            if run.stop_reason == 'startup_timeout':
                detail = (last_wait_reason or run.policy_input_status) if received else (
                    'No live snapshots received. Check that the game extension is connected '
                    'to ws://127.0.0.1:8765 and its AI loop is enabled.')
                print(f'Startup timed out: {detail}', flush=True)
        if ended_at is not None and time.monotonic() - ended_at >= drain_seconds:
            return run.snapshot()
        if not socket.poll(poll_ms, zmq.POLLIN):
            continue
        raw = socket.recv()
        received += 1
        request_id = None
        try:
            message = decode_message(raw)
            request_id = message.get('requestId') if isinstance(message, dict) else None
            if stop_event.is_set():
                run.stop('manual_stop')
            response = run.handle(message)
        except (ValueError, UnicodeDecodeError) as error:
            response = run._reply(request_id, reason=f'invalid_json: {error}')
        except Exception:
            run.stop('adapter_error', interrupted=True)
            response = run._reply(request_id, reason='adapter_error')
        run.check_deadlines()
        if run.terminal:
            response = run._reply(request_id)
        if stop_event.is_set():
            run.stop('manual_stop')
            response = run._reply(request_id)
        socket.send_json(response)
        if response.get('move') is not None and not announced_running:
            print('Live snapshots accepted; policy actions are flowing.', flush=True)
            announced_running = True
        if not announced_running and response.get('idleReason'):
            reason = response['idleReason']
            # One startup diagnostic; retain the latest reason for timeout output.
            if last_wait_reason is None:
                print(f'Waiting for usable live input: {reason}', flush=True)
                if reason == 'extension_update_required':
                    print('Update battler.js, content-bridge.js and background.js using '
                          'Automation/Bridge/MYSTIC_EXTENSION.md, then reload the extension '
                          'and refresh the game page. The bridge is receiving snapshots, '
                          'but they lack mystic_explicit_target_v1.', flush=True)
            last_wait_reason = reason


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--observe-only', action='store_true',
                        help='Observe lifecycle and inputs without sending gameplay commands')
    mode.add_argument('--checkpoint', type=Path, help='Explicit simulator checkpoint to load')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sampled', action='store_true', help='Sample actions instead of greedy inference')
    feedback = parser.add_mutually_exclusive_group()
    feedback.add_argument('--confirmed-feedback', action='store_true',
                          help='Require authoritative execution feedback for history')
    feedback.add_argument('--assume-sim-parity', action='store_true',
                          help='Estimate 80-input history from simulator rules; assumes cooldowns ready at launch')
    parser.add_argument('--input-size', type=int, choices=(56, 71, 80), default=None,
                        help='Observation mode input size; checkpoint mode infers its size (80 needs an explicit history mode)')
    parser.add_argument('--bind', default='tcp://127.0.0.1:5555')
    parser.add_argument('--startup-timeout', type=float, default=30.)
    parser.add_argument('--idle-timeout', type=float, default=2.)
    parser.add_argument('--acknowledgement-timeout', type=float, default=2.)
    parser.add_argument('--drain-seconds', type=float, default=1.)
    parser.add_argument('--snapshot-output', type=Path, help='New JSON file; defaults to a unique run directory')
    args = parser.parse_args(argv)
    if not 0 < args.drain_seconds <= 5:
        parser.error('--drain-seconds must be in (0, 5]')
    if args.observe_only and (args.sampled or args.confirmed_feedback or args.assume_sim_parity):
        parser.error('--sampled and feedback options require --checkpoint')
    actor = None
    try:
        if args.checkpoint:
            from .policy import FrozenPolicy
            actor = FrozenPolicy(args.checkpoint, device=args.device, sampled=args.sampled,
                                 assume_sim_parity=args.assume_sim_parity)
            if args.input_size is not None and args.input_size != actor.input_size:
                raise ValueError('--input-size conflicts with the checkpoint contract')
        adapter = actor.input_adapter if actor else LivePolicyInputs(args.input_size or 56)
        if actor and adapter.requires_execution_feedback and not args.confirmed_feedback:
            raise ValueError('This 80-input checkpoint requires --confirmed-feedback and an external '
                             'producer of execution outcomes and observationContext; the ordinary browser '
                             'does not supply them. Alternatively select --assume-sim-parity for '
                             'explicit simulator-estimated history.')
        progression = ('confirmed_feedback' if args.confirmed_feedback or adapter.requires_execution_feedback
                       else 'snapshot')
        from .commands import LiveCommands
        run = SingleRun(actor, input_adapter=adapter, progression=progression,
                        command_adapter=LiveCommands(adapter.encoder) if actor else None,
                        startup_timeout=args.startup_timeout, idle_timeout=args.idle_timeout,
                        acknowledgement_timeout=args.acknowledgement_timeout)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    output = args.snapshot_output or (Path('runs/mystic_live') /
             f'run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{run.run_id[:8]}' / 'final_state.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before binding. A launch never silently replaces earlier evidence.
    try:
        report_file = output.open('x', encoding='utf-8')
    except OSError as error:
        parser.error(f'Cannot reserve a new final snapshot: {error}')
    stop_event = threading.Event()
    previous_handler = signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(args.bind)
        description = (f'FROZEN ACTOR / {actor.checkpoint} / '
                       f'{"sampled" if actor.sampled else "greedy"} / {run.progression}'
                       if actor else 'OBSERVATION ONLY / no checkpoint / no gameplay commands')
        if args.assume_sim_parity:
            description += ' / SIMULATOR-ESTIMATED HISTORY / initial cooldowns assumed ready'
        print(f'Mystic live lifecycle listening on {args.bind}\nRun: {run.run_id}\n'
              f'{description}\n'
              f'Input contract: {run.input_adapter.schema}\nFinal state: {output}', flush=True)
        serve(run, socket, stop_event=stop_event, drain_seconds=args.drain_seconds)
    except Exception:
        run.stop('transport_error', interrupted=True)
        raise
    finally:
        if not run.terminal:
            run.stop('manual_stop')
        try:
            snapshot = run.snapshot()
            if actor is not None:
                snapshot['policy'] = dict(checkpoint=actor.checkpoint, sampled=actor.sampled,
                                          metadata=actor.metadata, progression=run.progression,
                                          assume_sim_parity=args.assume_sim_parity)
            json.dump(snapshot, report_file, indent=2, allow_nan=False)
            report_file.write('\n')
        finally:
            report_file.close()
            socket.close(linger=0)
            context.term()
            signal.signal(signal.SIGINT, previous_handler)
        print(f'Run {run.state.value}: {run.stop_reason}. Final state saved: {output}', flush=True)
    return 1 if run.state.value == 'interrupted' else 0


if __name__ == '__main__':
    raise SystemExit(main())
