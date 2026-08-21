"""Diagnose each hop in the WebSocket-to-ZeroMQ request path.

By default this starts the repository's mock ZeroMQ backend and real WebSocket
bridge, runs the probes, and stops both processes. Pass ``--external`` to probe
an already-running bridge and backend instead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import zmq
from websockets.sync.client import connect

DEFAULT_WS_URL = "ws://127.0.0.1:8765"
DEFAULT_ZMQ_URL = "tcp://127.0.0.1:5555"
DEFAULT_TIMEOUT_SECONDS = 7.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 10.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiagnosticError(RuntimeError):
    """A failure whose message identifies the stalled request stage."""


@dataclass(frozen=True)
class StageResult:
    name: str
    passed: bool
    elapsed_seconds: float
    detail: str


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    output_thread: threading.Thread


class StageTrace:
    def __init__(self, name: str) -> None:
        self.name = name
        self.started_at = time.monotonic()

    def checkpoint(self, detail: str) -> None:
        elapsed = time.monotonic() - self.started_at
        print(f"[{self.name} +{elapsed:0.3f}s] {detail}", flush=True)


def request_id(label: str) -> str:
    return f"diagnostic-{label}-{uuid.uuid4().hex[:12]}"


def ai_tick_payload(label: str) -> dict[str, Any]:
    """Return a small world state that is valid for the current Env16 parser."""
    return {
        "type": "ai_tick",
        "requestId": request_id(label),
        "timestamp": int(time.time() * 1000),
        "pageUrl": "diagnostic://ws-zmq",
        "worldState": {
            "timestamp": int(time.time() * 1000),
            "player": {
                "id": "diagnostic-player",
                "name": "Bridge diagnostic",
                "mapID": "map53",
                "mapX": 0,
                "mapY": 35,
                "direction": "up",
                "hp": 100,
                "maxHp": 100,
                "mp": 100,
                "maxMp": 100,
            },
            "entities": [],
        },
    }


def decode_object(raw_message: str | bytes, source: str) -> dict[str, Any]:
    try:
        message = json.loads(raw_message)
    except (json.JSONDecodeError, TypeError) as error:
        raise DiagnosticError(f"{source} returned invalid JSON: {raw_message!r}") from error

    if not isinstance(message, dict):
        raise DiagnosticError(
            f"{source} returned {type(message).__name__}, expected a JSON object"
        )
    return message


def validate_response(
    response: dict[str, Any],
    expected_request_id: str,
    source: str,
) -> str:
    actual_request_id = response.get("requestId")
    if actual_request_id != expected_request_id:
        raise DiagnosticError(
            f"{source} response requestId mismatch: "
            f"expected {expected_request_id!r}, got {actual_request_id!r}"
        )

    if response.get("error"):
        raise DiagnosticError(f"{source} returned error: {response['error']}")

    return (
        f"requestId={actual_request_id} "
        f"type={response.get('type')!r} move={response.get('move')!r}"
    )


def run_stage(
    name: str,
    operation: Callable[[StageTrace], str],
) -> StageResult:
    trace = StageTrace(name)
    trace.checkpoint("START")
    try:
        detail = operation(trace)
    except KeyboardInterrupt:
        raise
    except Exception as error:
        elapsed = time.monotonic() - trace.started_at
        detail = f"{type(error).__name__}: {error}"
        trace.checkpoint(f"FAIL - {detail}")
        return StageResult(name, False, elapsed, detail)

    elapsed = time.monotonic() - trace.started_at
    trace.checkpoint(f"PASS - {detail}")
    return StageResult(name, True, elapsed, detail)


def start_managed_process(
    trace: StageTrace,
    *,
    name: str,
    module: str,
    ready_signal: str,
    timeout_seconds: float,
) -> ManagedProcess:
    command = [sys.executable, "-u", "-m", module]
    trace.checkpoint(f"starting: {subprocess.list2cmdline(command)}")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    ready_event = threading.Event()
    output_finished = threading.Event()

    def stream_output() -> None:
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()
                if line:
                    print(f"[{name}] {line}", flush=True)
                if ready_signal in line:
                    ready_event.set()
        finally:
            output_finished.set()

    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()
    deadline = time.monotonic() + timeout_seconds

    try:
        while not ready_event.is_set():
            return_code = process.poll()
            if return_code is not None:
                output_finished.wait(timeout=0.2)
                raise DiagnosticError(
                    f"{name} exited with code {return_code} before emitting "
                    f"{ready_signal!r}; check the prefixed process output above"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiagnosticError(
                    f"{name} didn't emit {ready_signal!r} within "
                    f"{timeout_seconds:g}s"
                )
            ready_event.wait(timeout=min(0.05, remaining))
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        output_thread.join(timeout=1)
        raise

    trace.checkpoint(f"{name} READY pid={process.pid}")
    return ManagedProcess(name, process, output_thread)


def stop_managed_process(managed_process: ManagedProcess) -> None:
    process = managed_process.process
    if process.poll() is None:
        print(
            f"[cleanup] stopping {managed_process.name} pid={process.pid}",
            flush=True,
        )
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print(
                f"[cleanup] killing unresponsive {managed_process.name} "
                f"pid={process.pid}",
                flush=True,
            )
            process.kill()
            process.wait(timeout=3)
    managed_process.output_thread.join(timeout=1)


def websocket_round_trip(
    trace: StageTrace,
    *,
    ws_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    expected_type: str | None,
) -> dict[str, Any]:
    trace.checkpoint(f"opening WebSocket {ws_url}")
    try:
        with connect(
            ws_url,
            open_timeout=timeout_seconds,
            close_timeout=1,
            ping_interval=None,
            proxy=None,
        ) as websocket:
            trace.checkpoint("WebSocket OPEN")
            websocket.send(json.dumps(payload))
            trace.checkpoint(
                f"WebSocket SENT requestId={payload.get('requestId')} "
                f"type={payload.get('type')}"
            )
            try:
                raw_response = websocket.recv(timeout=timeout_seconds)
            except TimeoutError as error:
                raise DiagnosticError(
                    f"WebSocket receive timed out after {timeout_seconds:g}s"
                ) from error
            trace.checkpoint("WebSocket RECEIVED a frame")
    except DiagnosticError:
        raise
    except TimeoutError as error:
        raise DiagnosticError(
            f"WebSocket open timed out after {timeout_seconds:g}s at {ws_url}"
        ) from error
    except OSError as error:
        raise DiagnosticError(f"WebSocket connect failed at {ws_url}: {error}") from error

    response = decode_object(raw_response, "WebSocket bridge")
    if expected_type is not None and response.get("type") != expected_type:
        raise DiagnosticError(
            "WebSocket bridge returned unexpected type: "
            f"expected {expected_type!r}, got {response.get('type')!r}; "
            f"response={response!r}"
        )
    return response


def check_websocket_ping(
    trace: StageTrace,
    *,
    ws_url: str,
    timeout_seconds: float,
) -> str:
    ping_request_id = request_id("ping")
    response = websocket_round_trip(
        trace,
        ws_url=ws_url,
        payload={
            "type": "ping",
            "requestId": ping_request_id,
            "timestamp": int(time.time() * 1000),
        },
        timeout_seconds=timeout_seconds,
        expected_type="pong",
    )
    return validate_response(response, ping_request_id, "WebSocket ping")


def check_direct_zmq(
    trace: StageTrace,
    *,
    zmq_url: str,
    timeout_seconds: float,
) -> str:
    timeout_ms = max(1, int(timeout_seconds * 1000))
    payload = ai_tick_payload("zmq")
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.IMMEDIATE, 1)

    try:
        trace.checkpoint(f"configuring ZeroMQ REQ connection to {zmq_url}")
        socket.connect(zmq_url)

        trace.checkpoint("waiting for a connected/writable ZeroMQ peer")
        events = socket.poll(timeout_ms, zmq.POLLOUT)
        if not events & zmq.POLLOUT:
            raise DiagnosticError(
                f"ZeroMQ connect/send timed out after {timeout_seconds:g}s; "
                f"no writable REP peer at {zmq_url}"
            )

        try:
            socket.send_json(payload, flags=zmq.NOBLOCK)
        except zmq.Again as error:
            raise DiagnosticError(
                "ZeroMQ peer became unavailable before the request could be sent"
            ) from error
        trace.checkpoint(f"ZeroMQ SENT requestId={payload['requestId']}")

        events = socket.poll(timeout_ms, zmq.POLLIN)
        if not events & zmq.POLLIN:
            raise DiagnosticError(
                f"ZeroMQ receive timed out after {timeout_seconds:g}s; "
                "the request was sent but the REP backend didn't reply"
            )

        try:
            response = socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again as error:
            raise DiagnosticError(
                "ZeroMQ reported a reply but it wasn't readable"
            ) from error
        trace.checkpoint("ZeroMQ RECEIVED a reply")
    finally:
        socket.close(linger=0)

    if not isinstance(response, dict):
        raise DiagnosticError(
            f"ZeroMQ backend returned {type(response).__name__}, expected a JSON object"
        )
    return validate_response(response, payload["requestId"], "ZeroMQ backend")


def check_end_to_end(
    trace: StageTrace,
    *,
    ws_url: str,
    timeout_seconds: float,
    sequence: int,
) -> str:
    payload = ai_tick_payload(f"e2e-{sequence}")
    response = websocket_round_trip(
        trace,
        ws_url=ws_url,
        payload=payload,
        timeout_seconds=timeout_seconds,
        expected_type=None,
    )
    return validate_response(response, payload["requestId"], "End-to-end bridge")


def print_summary(
    results: list[StageResult],
    *,
    direct_zmq_was_skipped: bool,
) -> None:
    print("\n=== Diagnostic summary ===", flush=True)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status:4}  {result.name:<22} "
            f"{result.elapsed_seconds:7.3f}s  {result.detail}",
            flush=True,
        )

    ping_result = next(
        (result for result in results if result.name == "WebSocket ping"),
        None,
    )
    zmq_result = next(
        (result for result in results if result.name == "Direct ZeroMQ"),
        None,
    )
    e2e_results = [
        result for result in results if result.name.startswith("End-to-end")
    ]
    startup_failure = next(
        (
            result
            for result in results
            if result.name.endswith("startup") and not result.passed
        ),
        None,
    )

    print("\nLikely hang location:", flush=True)
    if startup_failure is not None:
        print(
            f"- {startup_failure.name}: the self-contained test fixture "
            "couldn't start. Check its prefixed output above.",
            flush=True,
        )
    elif ping_result is not None and not ping_result.passed:
        print(
            "- WebSocket listener/handler: the bridge didn't complete a ping "
            "that bypasses ZeroMQ.",
            flush=True,
        )
    elif (
        zmq_result is not None
        and not zmq_result.passed
        and any(not result.passed for result in e2e_results)
    ):
        print(
            "- ZeroMQ backend: WebSocket ping works, but the REP peer isn't "
            "available or isn't replying.",
            flush=True,
        )
    elif (
        zmq_result is not None
        and zmq_result.passed
        and any(not result.passed for result in e2e_results)
    ):
        print(
            "- Bridge forwarding/correlation: WebSocket and ZeroMQ work "
            "separately, but the combined ai_tick path fails.",
            flush=True,
        )
    elif e2e_results and all(result.passed for result in e2e_results):
        if zmq_result is not None and not zmq_result.passed:
            print(
                "- The end-to-end path works. The independent ZMQ probe failed "
                "to attach, so inspect backend connection routing if that probe "
                "matters.",
                flush=True,
            )
        else:
            print("- None observed; every requested stage passed.", flush=True)
    elif direct_zmq_was_skipped:
        print(
            "- Indeterminate because the direct ZMQ probe was skipped; rerun "
            "without --skip-zmq to isolate the backend.",
            flush=True,
        )
    else:
        print("- See the first failed checkpoint above.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pinpoint a hang in the WebSocket-to-ZeroMQ bridge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--zmq-url", default=DEFAULT_ZMQ_URL)
    parser.add_argument(
        "--external",
        action="store_true",
        help=(
            "Probe an already-running bridge/backend instead of starting the "
            "mock backend and real bridge automatically."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="Deadline for each self-contained fixture process to become ready.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Deadline for each connect or receive checkpoint in seconds.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of end-to-end ai_tick round trips.",
    )
    parser.add_argument(
        "--skip-zmq",
        action="store_true",
        help="Skip the independent direct-ZeroMQ probe.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be greater than zero")
    if args.count <= 0:
        parser.error("--count must be greater than zero")
    if not args.external and (
        args.ws_url != DEFAULT_WS_URL or args.zmq_url != DEFAULT_ZMQ_URL
    ):
        parser.error("endpoint overrides require --external")
    return args


def main() -> int:
    args = parse_args()
    print("WebSocket/ZeroMQ staged diagnostic", flush=True)
    print(
        "Mode:               "
        + ("external services" if args.external else "self-contained mock stack"),
        flush=True,
    )
    print(f"WebSocket endpoint: {args.ws_url}", flush=True)
    print(f"ZeroMQ endpoint:   {args.zmq_url}", flush=True)
    print(f"Stage timeout:     {args.timeout:g}s", flush=True)
    if args.external and not args.skip_zmq:
        print(
            "NOTE: the direct ZMQ probe and each end-to-end probe send one "
            "ai_tick to the active backend.",
            flush=True,
        )

    results: list[StageResult] = []
    managed_processes: list[ManagedProcess] = []

    try:
        if not args.external:
            def start_mock_backend(trace: StageTrace) -> str:
                managed_process = start_managed_process(
                    trace,
                    name="mock-zmq",
                    module="Tests.zmq_ai_backend",
                    ready_signal="ZeroMQ AI backend listening",
                    timeout_seconds=args.startup_timeout,
                )
                managed_processes.append(managed_process)
                return f"pid={managed_process.process.pid} endpoint={args.zmq_url}"

            mock_startup = run_stage("Mock ZMQ startup", start_mock_backend)
            results.append(mock_startup)
            if not mock_startup.passed:
                print_summary(results, direct_zmq_was_skipped=args.skip_zmq)
                return 1

            def start_bridge(trace: StageTrace) -> str:
                managed_process = start_managed_process(
                    trace,
                    name="ws-bridge",
                    module="Automation.Bridge.ws_zmq_bridge",
                    ready_signal="WebSocket bridge listening",
                    timeout_seconds=args.startup_timeout,
                )
                managed_processes.append(managed_process)
                return f"pid={managed_process.process.pid} endpoint={args.ws_url}"

            bridge_startup = run_stage("WebSocket bridge startup", start_bridge)
            results.append(bridge_startup)
            if not bridge_startup.passed:
                print_summary(results, direct_zmq_was_skipped=args.skip_zmq)
                return 1

        results.append(
            run_stage(
                "WebSocket ping",
                lambda trace: check_websocket_ping(
                    trace,
                    ws_url=args.ws_url,
                    timeout_seconds=args.timeout,
                ),
            )
        )

        if not args.skip_zmq:
            results.append(
                run_stage(
                    "Direct ZeroMQ",
                    lambda trace: check_direct_zmq(
                        trace,
                        zmq_url=args.zmq_url,
                        timeout_seconds=args.timeout,
                    ),
                )
            )

        for sequence in range(1, args.count + 1):
            stage_name = f"End-to-end #{sequence}"
            results.append(
                run_stage(
                    stage_name,
                    lambda trace, sequence=sequence: check_end_to_end(
                        trace,
                        ws_url=args.ws_url,
                        timeout_seconds=args.timeout,
                        sequence=sequence,
                    ),
                )
            )

        print_summary(results, direct_zmq_was_skipped=args.skip_zmq)
        return 0 if all(result.passed for result in results) else 1
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted.", flush=True)
        return 130
    finally:
        for managed_process in reversed(managed_processes):
            stop_managed_process(managed_process)


if __name__ == "__main__":
    raise SystemExit(main())
