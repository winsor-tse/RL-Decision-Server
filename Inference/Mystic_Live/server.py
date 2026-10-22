"""Phase 8B REP server: one observation-only lifecycle per explicit launch.

No checkpoint is selected or inferred. Real actor loading/encoding belongs to
8C/8F. Run the matching bridge with --single-run for connection identity/events.
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
    while True:
        if stop_event.is_set():
            run.stop('manual_stop')
        run.check_deadlines()
        if run.terminal and ended_at is None:
            ended_at = time.monotonic()
        if ended_at is not None and time.monotonic() - ended_at >= drain_seconds:
            return run.snapshot()
        if not socket.poll(poll_ms, zmq.POLLIN):
            continue
        raw = socket.recv()
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe-only', action='store_true', required=True,
                        help='Required in 8B: observe lifecycle without sending gameplay commands')
    parser.add_argument('--bind', default='tcp://127.0.0.1:5555')
    parser.add_argument('--startup-timeout', type=float, default=30.)
    parser.add_argument('--idle-timeout', type=float, default=2.)
    parser.add_argument('--acknowledgement-timeout', type=float, default=2.)
    parser.add_argument('--drain-seconds', type=float, default=1.)
    parser.add_argument('--snapshot-output', type=Path, help='New JSON file; defaults to a unique run directory')
    args = parser.parse_args(argv)
    if not 0 < args.drain_seconds <= 5:
        parser.error('--drain-seconds must be in (0, 5]')
    try:
        run = SingleRun(startup_timeout=args.startup_timeout, idle_timeout=args.idle_timeout,
                        acknowledgement_timeout=args.acknowledgement_timeout)
    except ValueError as error:
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
        print(f'Mystic live lifecycle listening on {args.bind}\nRun: {run.run_id}\n'
              f'OBSERVATION ONLY / no checkpoint / no gameplay commands\nFinal state: {output}', flush=True)
        serve(run, socket, stop_event=stop_event, drain_seconds=args.drain_seconds)
    except Exception:
        run.stop('transport_error', interrupted=True)
        raise
    finally:
        if not run.terminal:
            run.stop('manual_stop')
        try:
            json.dump(run.snapshot(), report_file, indent=2, allow_nan=False)
            report_file.write('\n')
        finally:
            report_file.close()
            socket.close(linger=0)
            context.term()
            signal.signal(signal.SIGINT, previous_handler)
        print(f'Run {run.state.value}: {run.stop_reason}. Final state saved: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
