"""Record live game states and player input for offline reinforcement learning.

This module is an alternative ZeroMQ backend for the existing WebSocket bridge.
It always replies to ``ai_tick`` messages with a configurable no-op action and
only writes frames while a CLI recording session is active.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol


DEFAULT_BIND_URL = "tcp://127.0.0.1:5555"
DEFAULT_DATABASE = Path(__file__).with_name("recordings.sqlite3")


def _utc_iso(timestamp_ns: int | None = None) -> str:
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()
    return datetime.fromtimestamp(
        timestamp_ns / 1_000_000_000,
        tz=timezone.utc,
    ).isoformat(timespec="microseconds")


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class InputMonitor(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def reset(self) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


class NullInputMonitor:
    """Input source used when states will be annotated manually."""

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "capture_available": False,
            "keys_down": [],
            "events": [],
        }


_VK_NAMES = {
    0x01: "MOUSE_LEFT",
    0x02: "MOUSE_RIGHT",
    0x04: "MOUSE_MIDDLE",
    0x05: "MOUSE_X1",
    0x06: "MOUSE_X2",
    0x08: "BACKSPACE",
    0x09: "TAB",
    0x0D: "ENTER",
    0x13: "PAUSE",
    0x14: "CAPS_LOCK",
    0x1B: "ESCAPE",
    0x20: "SPACE",
    0x21: "PAGE_UP",
    0x22: "PAGE_DOWN",
    0x23: "END",
    0x24: "HOME",
    0x25: "ARROW_LEFT",
    0x26: "ARROW_UP",
    0x27: "ARROW_RIGHT",
    0x28: "ARROW_DOWN",
    0x2C: "PRINT_SCREEN",
    0x2D: "INSERT",
    0x2E: "DELETE",
    0x5B: "LEFT_WINDOWS",
    0x5C: "RIGHT_WINDOWS",
    0x60: "NUMPAD_0",
    0x61: "NUMPAD_1",
    0x62: "NUMPAD_2",
    0x63: "NUMPAD_3",
    0x64: "NUMPAD_4",
    0x65: "NUMPAD_5",
    0x66: "NUMPAD_6",
    0x67: "NUMPAD_7",
    0x68: "NUMPAD_8",
    0x69: "NUMPAD_9",
    0x6A: "NUMPAD_MULTIPLY",
    0x6B: "NUMPAD_ADD",
    0x6D: "NUMPAD_SUBTRACT",
    0x6E: "NUMPAD_DECIMAL",
    0x6F: "NUMPAD_DIVIDE",
    0x90: "NUM_LOCK",
    0x91: "SCROLL_LOCK",
    0xA0: "SHIFT_LEFT",
    0xA1: "SHIFT_RIGHT",
    0xA2: "CTRL_LEFT",
    0xA3: "CTRL_RIGHT",
    0xA4: "ALT_LEFT",
    0xA5: "ALT_RIGHT",
    0xBA: ";",
    0xBB: "=",
    0xBC: ",",
    0xBD: "-",
    0xBE: ".",
    0xBF: "/",
    0xC0: "`",
    0xDB: "[",
    0xDC: "\\",
    0xDD: "]",
    0xDE: "'",
}


def virtual_key_name(virtual_key: int) -> str:
    if 0x30 <= virtual_key <= 0x39 or 0x41 <= virtual_key <= 0x5A:
        return chr(virtual_key)
    if 0x70 <= virtual_key <= 0x87:
        return f"F{virtual_key - 0x6F}"
    return _VK_NAMES.get(virtual_key, f"VK_{virtual_key:02X}")


class WindowsInputMonitor:
    """Capture global Windows key/button state without extra dependencies."""

    # Generic SHIFT/CTRL/ALT duplicate their left/right virtual keys.
    _VIRTUAL_KEYS = tuple(
        virtual_key
        for virtual_key in range(1, 255)
        if virtual_key not in (0x10, 0x11, 0x12)
    )

    def __init__(self, poll_interval_seconds: float = 0.005):
        if os.name != "nt":
            raise RuntimeError(
                "Global key capture is currently supported on Windows only. "
                "Use --no-keyboard to record states for manual annotation."
            )
        if poll_interval_seconds <= 0:
            raise ValueError("Keyboard poll interval must be greater than zero.")

        self._poll_interval_seconds = poll_interval_seconds
        self._get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        self._get_async_key_state.argtypes = [ctypes.c_int]
        self._get_async_key_state.restype = ctypes.c_short
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._keys_down: set[int] = set()
        self._events: list[dict[str, Any]] = []

    def _read_keys_down(self) -> set[int]:
        return {
            virtual_key
            for virtual_key in self._VIRTUAL_KEYS
            if self._get_async_key_state(virtual_key) & 0x8000
        }

    @staticmethod
    def _key(virtual_key: int) -> dict[str, Any]:
        return {
            "key": virtual_key_name(virtual_key),
            "vk": virtual_key,
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._keys_down = self._read_keys_down()
            self._events.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="offline-input-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval_seconds):
            current_keys = self._read_keys_down()
            timestamp_ns = time.time_ns()
            with self._lock:
                pressed = current_keys - self._keys_down
                released = self._keys_down - current_keys
                for virtual_key in sorted(pressed):
                    self._events.append(
                        {
                            **self._key(virtual_key),
                            "event": "key_down",
                            "timestamp_unix_ns": timestamp_ns,
                        }
                    )
                for virtual_key in sorted(released):
                    self._events.append(
                        {
                            **self._key(virtual_key),
                            "event": "key_up",
                            "timestamp_unix_ns": timestamp_ns,
                        }
                    )
                self._keys_down = current_keys

    def reset(self) -> None:
        current_keys = self._read_keys_down()
        with self._lock:
            self._keys_down = current_keys
            self._events.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = {
                "capture_available": True,
                "keys_down": [
                    self._key(virtual_key)
                    for virtual_key in sorted(self._keys_down)
                ],
                "events": list(self._events),
            }
            self._events.clear()
        return snapshot

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None


class RecordingStore:
    """Thread-safe SQLite storage for recording sessions and frames."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=10,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS recording_sessions (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    started_at_utc TEXT NOT NULL,
                    started_at_unix_ns INTEGER NOT NULL,
                    ended_at_utc TEXT,
                    ended_at_unix_ns INTEGER
                );

                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    received_at_utc TEXT NOT NULL,
                    received_at_unix_ns INTEGER NOT NULL,
                    request_id_json TEXT,
                    payload_json TEXT NOT NULL,
                    world_state_json TEXT,
                    input_json TEXT NOT NULL,
                    action_label TEXT,
                    action_source TEXT,
                    response_json TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES recording_sessions(id),
                    UNIQUE (session_id, sequence_number)
                );

                CREATE INDEX IF NOT EXISTS frames_session_id_index
                ON frames(session_id, sequence_number);

                PRAGMA user_version = 1;
                """
            )
            self._connection.commit()

    def create_session(self, name: str) -> int:
        timestamp_ns = time.time_ns()
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO recording_sessions (
                    name, started_at_utc, started_at_unix_ns
                ) VALUES (?, ?, ?)
                """,
                (name, _utc_iso(timestamp_ns), timestamp_ns),
            )
            self._connection.commit()
            return int(cursor.lastrowid)

    def finish_session(self, session_id: int) -> None:
        timestamp_ns = time.time_ns()
        with self._lock:
            self._connection.execute(
                """
                UPDATE recording_sessions
                SET ended_at_utc = ?, ended_at_unix_ns = ?
                WHERE id = ? AND ended_at_utc IS NULL
                """,
                (_utc_iso(timestamp_ns), timestamp_ns, session_id),
            )
            self._connection.commit()

    def insert_frame(
        self,
        *,
        session_id: int,
        sequence_number: int,
        received_at_ns: int,
        message: dict[str, Any],
        input_snapshot: dict[str, Any],
        response: dict[str, Any],
    ) -> int:
        request_id = message.get("requestId")
        world_state = message.get("worldState")
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO frames (
                    session_id,
                    sequence_number,
                    received_at_utc,
                    received_at_unix_ns,
                    request_id_json,
                    payload_json,
                    world_state_json,
                    input_json,
                    response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    sequence_number,
                    _utc_iso(received_at_ns),
                    received_at_ns,
                    _json(request_id) if request_id is not None else None,
                    _json(message),
                    _json(world_state) if world_state is not None else None,
                    _json(input_snapshot),
                    _json(response),
                ),
            )
            self._connection.commit()
            return int(cursor.lastrowid)

    def annotate_frame(self, frame_id: int, action_label: str) -> None:
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE frames
                SET action_label = ?, action_source = 'manual'
                WHERE id = ?
                """,
                (action_label, frame_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Frame {frame_id} does not exist.")
            self._connection.commit()

    def count_frames(self, session_id: int) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM frames WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["count"])

    def list_sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self._connection.execute(
                    """
                    SELECT
                        recording_sessions.*,
                        COUNT(frames.id) AS frame_count
                    FROM recording_sessions
                    LEFT JOIN frames ON frames.session_id = recording_sessions.id
                    GROUP BY recording_sessions.id
                    ORDER BY recording_sessions.id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )

    def list_recent_frames(self, limit: int = 10) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self._connection.execute(
                    """
                    SELECT
                        id,
                        session_id,
                        sequence_number,
                        received_at_utc,
                        input_json,
                        action_label
                    FROM frames
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class RecorderController:
    """Coordinate recording state, input snapshots, and database writes."""

    def __init__(
        self,
        store: RecordingStore,
        input_monitor: InputMonitor,
        no_op_action: Any = "NoOp",
    ):
        self.store = store
        self.input_monitor = input_monitor
        self.no_op_action = no_op_action
        self._lock = threading.RLock()
        self._session_id: int | None = None
        self._session_name: str | None = None
        self._sequence_number = 0

    def start_recording(self, name: str | None = None) -> int:
        with self._lock:
            if self._session_id is not None:
                raise RuntimeError(
                    f"Session {self._session_id} is already recording."
                )
            if not name:
                name = datetime.now().strftime("recording-%Y%m%d-%H%M%S")
            self.input_monitor.reset()
            self._session_id = self.store.create_session(name)
            self._session_name = name
            self._sequence_number = 0
            return self._session_id

    def stop_recording(self) -> tuple[int, int]:
        with self._lock:
            if self._session_id is None:
                raise RuntimeError("No recording is active.")
            session_id = self._session_id
            frame_count = self._sequence_number
            self.store.finish_session(session_id)
            self._session_id = None
            self._session_name = None
            self._sequence_number = 0
            return session_id, frame_count

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "recording": self._session_id is not None,
                "session_id": self._session_id,
                "session_name": self._session_name,
                "frame_count": self._sequence_number,
            }

    def handle_message(self, message: Any) -> dict[str, Any]:
        if not isinstance(message, dict):
            return {
                "type": "error",
                "requestId": None,
                "error": "ZeroMQ payload must be a JSON object.",
            }

        message_type = message.get("type")
        if message_type != "ai_tick":
            return {
                "type": "error",
                "requestId": message.get("requestId"),
                "error": f"Unknown message type: {message_type}",
            }

        received_at_ns = time.time_ns()
        response = {
            "type": "ai_result",
            "requestId": message.get("requestId"),
            "move": self.no_op_action,
            "reset": False,
            "serverTime": time.time(),
        }

        with self._lock:
            if self._session_id is not None:
                self._sequence_number += 1
                self.store.insert_frame(
                    session_id=self._session_id,
                    sequence_number=self._sequence_number,
                    received_at_ns=received_at_ns,
                    message=message,
                    input_snapshot=self.input_monitor.snapshot(),
                    response=response,
                )

        return response


class RecorderServer:
    """Background ZeroMQ REP server used while the CLI reads commands."""

    def __init__(
        self,
        bind_url: str,
        controller: RecorderController,
        report_error: Callable[[str], None] = print,
    ):
        self.bind_url = bind_url
        self.controller = controller
        self.report_error = report_error
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None

    def start(self, timeout_seconds: float = 5) -> None:
        self._stop_event.clear()
        self._ready_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._serve,
            name="offline-zmq-recorder",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout_seconds):
            raise TimeoutError(f"Timed out binding recorder to {self.bind_url}.")
        if self._startup_error is not None:
            raise RuntimeError(
                f"Could not bind recorder to {self.bind_url}: "
                f"{self._startup_error}"
            ) from self._startup_error

    def _serve(self) -> None:
        socket = None
        try:
            import zmq

            context = zmq.Context.instance()
            socket = context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(self.bind_url)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            self._ready_event.set()

            while not self._stop_event.is_set():
                events = dict(poller.poll(timeout=100))
                if socket not in events:
                    continue
                try:
                    message = socket.recv_json()
                    response = self.controller.handle_message(message)
                except Exception as error:
                    self.report_error(f"Recorder error: {error}")
                    response = {
                        "type": "ai_result",
                        "requestId": None,
                        "move": self.controller.no_op_action,
                        "reset": False,
                        "serverTime": time.time(),
                        "error": str(error),
                    }
                socket.send_json(response)
        except BaseException as error:
            self._startup_error = error
            self._ready_event.set()
        finally:
            if socket is not None:
                socket.close(linger=0)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None


HELP_TEXT = """Commands:
  start [name]              start a new recording session
  stop                      stop the active session
  status                    show current recording status
  sessions [count]          show recent sessions
  frames [count]            show recent recorded frames and held keys
  annotate <frame> <action> set a frame's mapped action label/index
  help                      show this command list
  quit                      stop recording and exit
"""


def _positive_count(value: str, default: int) -> int:
    if not value:
        return default
    count = int(value)
    if count <= 0:
        raise ValueError("Count must be greater than zero.")
    return count


def _show_status(controller: RecorderController) -> None:
    status = controller.status()
    if status["recording"]:
        print(
            f"Recording session {status['session_id']} "
            f"({status['session_name']!r}): {status['frame_count']} frames"
        )
    else:
        print("Recorder is idle; incoming ticks receive no-op but are not saved.")


def _show_sessions(store: RecordingStore, limit: int) -> None:
    rows = store.list_sessions(limit)
    if not rows:
        print("No recording sessions yet.")
        return
    for row in rows:
        state = "recording" if row["ended_at_utc"] is None else "stopped"
        print(
            f"{row['id']:>5}  {row['frame_count']:>7} frames  "
            f"{state:<9}  {row['name']}"
        )


def _show_frames(store: RecordingStore, limit: int) -> None:
    rows = store.list_recent_frames(limit)
    if not rows:
        print("No frames recorded yet.")
        return
    for row in reversed(rows):
        input_snapshot = json.loads(row["input_json"])
        keys = "+".join(key["key"] for key in input_snapshot["keys_down"])
        print(
            f"frame={row['id']} session={row['session_id']} "
            f"seq={row['sequence_number']} keys={keys or '-'} "
            f"action={row['action_label'] or '-'}"
        )


def run_command_shell(
    controller: RecorderController,
    store: RecordingStore,
) -> None:
    print(HELP_TEXT)
    while True:
        try:
            raw_command = input("recorder> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw_command:
            continue

        try:
            parts = shlex.split(raw_command)
            command = parts[0].lower()
            arguments = parts[1:]

            if command == "start":
                name = " ".join(arguments) or None
                session_id = controller.start_recording(name)
                print(f"Recording session {session_id} started.")
            elif command == "stop":
                session_id, frame_count = controller.stop_recording()
                print(
                    f"Recording session {session_id} stopped "
                    f"with {frame_count} frames."
                )
            elif command == "status":
                _show_status(controller)
            elif command == "sessions":
                limit = _positive_count(arguments[0] if arguments else "", 20)
                _show_sessions(store, limit)
            elif command == "frames":
                limit = _positive_count(arguments[0] if arguments else "", 10)
                _show_frames(store, limit)
            elif command == "annotate":
                if len(arguments) < 2:
                    raise ValueError("Usage: annotate <frame> <action>")
                frame_id = int(arguments[0])
                action_label = " ".join(arguments[1:])
                store.annotate_frame(frame_id, action_label)
                print(f"Frame {frame_id} action set to {action_label!r}.")
            elif command == "help":
                print(HELP_TEXT)
            elif command in {"quit", "exit"}:
                return
            else:
                print(f"Unknown command: {command}. Enter 'help' for commands.")
        except (RuntimeError, ValueError) as error:
            print(f"Error: {error}")


def _parse_no_op_action(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record full game states and global player input to SQLite."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"SQLite output path (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--bind-url",
        default=DEFAULT_BIND_URL,
        help=f"ZeroMQ REP endpoint (default: {DEFAULT_BIND_URL})",
    )
    parser.add_argument(
        "--no-op-action",
        type=_parse_no_op_action,
        default="NoOp",
        help='JSON value returned as move on every tick (default: "NoOp")',
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable global input capture and annotate actions manually",
    )
    parser.add_argument(
        "--keyboard-poll-ms",
        type=float,
        default=5.0,
        help="Global input polling interval in milliseconds (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.keyboard_poll_ms <= 0:
        raise SystemExit("--keyboard-poll-ms must be greater than zero")

    input_monitor: InputMonitor
    if args.no_keyboard:
        input_monitor = NullInputMonitor()
    else:
        input_monitor = WindowsInputMonitor(args.keyboard_poll_ms / 1000)

    store = RecordingStore(args.database)
    controller = RecorderController(store, input_monitor, args.no_op_action)
    server = RecorderServer(args.bind_url, controller)

    input_monitor.start()
    try:
        server.start()
        print(f"Recorder backend listening at {args.bind_url}")
        print(f"SQLite database: {store.path}")
        print(f"No-op move sent to the game: {args.no_op_action!r}")
        run_command_shell(controller, store)
    finally:
        if controller.status()["recording"]:
            session_id, frame_count = controller.stop_recording()
            print(
                f"Recording session {session_id} closed with "
                f"{frame_count} frames."
            )
        server.stop()
        input_monitor.stop()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
