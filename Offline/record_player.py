"""Record player demonstrations from the live Yugen Saga socket.

The loop intentionally mirrors ``Env16.step``:

1. receive one ``ai_tick`` with ``socket.recv_json()``;
2. read ``message["worldState"]``;
3. snapshot the player's current keyboard input;
4. reply with the ``NoOp`` move;
5. save the state/action pair to SQLite.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Custom_enviornments.Load_env_config import load_env_config


DEFAULT_BIND_URL = str(load_env_config()["ZMQ_BIND_URL"])
DEFAULT_DATABASE = Path(__file__).with_name("recordings.sqlite3")


def utc_iso(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ns / 1_000_000_000,
        tz=timezone.utc,
    ).isoformat(timespec="microseconds")


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


VK_NAMES = {
    0x01: "MOUSE_LEFT",
    0x02: "MOUSE_RIGHT",
    0x04: "MOUSE_MIDDLE",
    0x05: "MOUSE_X1",
    0x06: "MOUSE_X2",
    0x08: "BACKSPACE",
    0x09: "TAB",
    0x0D: "ENTER",
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
    0x2D: "INSERT",
    0x2E: "DELETE",
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
    if 0x60 <= virtual_key <= 0x69:
        return f"NUMPAD{virtual_key - 0x60}"
    if 0x70 <= virtual_key <= 0x87:
        return f"F{virtual_key - 0x6F}"
    return VK_NAMES.get(virtual_key, f"VK_{virtual_key:02X}")


class NoInputCapture:
    """Return an empty action when data will be labeled manually."""

    def reset(self) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "capture_available": False,
            "keys_down": [],
            "events": [],
        }


class WindowsInputCapture:
    """Snapshot globally held Windows keys when each state arrives."""

    # Generic SHIFT/CTRL/ALT duplicate their left/right virtual keys.
    VIRTUAL_KEYS = tuple(
        virtual_key
        for virtual_key in range(1, 255)
        if virtual_key not in (0x10, 0x11, 0x12)
    )

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError(
                "Global input capture is supported on Windows only. "
                "Use --no-keyboard for manual action labeling."
            )
        self.get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        self.get_async_key_state.argtypes = [ctypes.c_int]
        self.get_async_key_state.restype = ctypes.c_short
        self.previous_keys: set[int] = set()
        self.reset()

    def read_states(self) -> dict[int, int]:
        return {
            virtual_key: int(self.get_async_key_state(virtual_key))
            for virtual_key in self.VIRTUAL_KEYS
        }

    @staticmethod
    def describe_key(virtual_key: int) -> dict[str, Any]:
        return {
            "key": virtual_key_name(virtual_key),
            "vk": virtual_key,
        }

    def reset(self) -> None:
        states = self.read_states()
        self.previous_keys = {
            virtual_key
            for virtual_key, state in states.items()
            if state & 0x8000
        }

    def snapshot(self) -> dict[str, Any]:
        states = self.read_states()
        current_keys = {
            virtual_key
            for virtual_key, state in states.items()
            if state & 0x8000
        }
        pressed_keys = (current_keys - self.previous_keys) | {
            virtual_key
            for virtual_key, state in states.items()
            if state & 0x0001
        }
        released_keys = self.previous_keys - current_keys
        timestamp_ns = time.time_ns()

        events = [
            {
                **self.describe_key(virtual_key),
                "event": "key_down",
                "timestamp_unix_ns": timestamp_ns,
            }
            for virtual_key in sorted(pressed_keys)
        ]
        events.extend(
            {
                **self.describe_key(virtual_key),
                "event": "key_up",
                "timestamp_unix_ns": timestamp_ns,
            }
            for virtual_key in sorted(released_keys)
        )
        self.previous_keys = current_keys

        return {
            "capture_available": True,
            "keys_down": [
                self.describe_key(virtual_key)
                for virtual_key in sorted(current_keys)
            ],
            "events": events,
        }


class RecordingDatabase:
    """SQLite storage used by the synchronous recording loop."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=10)
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 10000")
        self.connection.executescript(
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
                world_state_json TEXT NOT NULL,
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
        self.connection.commit()

    def start_session(self, name: str) -> int:
        timestamp_ns = time.time_ns()
        cursor = self.connection.execute(
            """
            INSERT INTO recording_sessions (
                name, started_at_utc, started_at_unix_ns
            ) VALUES (?, ?, ?)
            """,
            (name, utc_iso(timestamp_ns), timestamp_ns),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_session(self, session_id: int) -> None:
        timestamp_ns = time.time_ns()
        self.connection.execute(
            """
            UPDATE recording_sessions
            SET ended_at_utc = ?, ended_at_unix_ns = ?
            WHERE id = ? AND ended_at_utc IS NULL
            """,
            (utc_iso(timestamp_ns), timestamp_ns, session_id),
        )
        self.connection.commit()

    def save_frame(
        self,
        *,
        session_id: int,
        sequence_number: int,
        received_at_ns: int,
        message: dict[str, Any],
        world_state: Any,
        player_input: dict[str, Any],
        response: dict[str, Any],
    ) -> int:
        request_id = message.get("requestId")
        cursor = self.connection.execute(
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
                utc_iso(received_at_ns),
                received_at_ns,
                json_text(request_id) if request_id is not None else None,
                json_text(message),
                json_text(world_state),
                json_text(player_input),
                json_text(response),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def close(self) -> None:
        self.connection.close()


class PlayerRecorder:
    """Receive and record one state/action pair per game ``ai_tick``."""

    def __init__(
        self,
        database_path: str | Path,
        bind_url: str = DEFAULT_BIND_URL,
        no_op_action: Any = "NoOp",
        input_capture: Any | None = None,
        socket: Any | None = None,
    ):
        self.database = RecordingDatabase(database_path)
        self.no_op_action = no_op_action
        self.input_capture = input_capture or WindowsInputCapture()
        self.session_id: int | None = None
        self.sequence_number = 0
        self.owns_socket = socket is None

        if socket is None:
            import zmq

            self.context = zmq.Context.instance()
            self.socket = self.context.socket(zmq.REP)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.bind(bind_url)
        else:
            self.context = None
            self.socket = socket

    def build_response(self, message: Any) -> dict[str, Any]:
        if not isinstance(message, dict):
            return {
                "type": "error",
                "requestId": None,
                "error": "ZeroMQ payload must be a JSON object.",
            }
        message_type = message.get("type")
        if message_type == "ai_tick":
            return {
                "type": "ai_result",
                "requestId": message.get("requestId"),
                "move": self.no_op_action,
                "reset": False,
                "serverTime": time.time(),
            }
        return {
            "type": "error",
            "requestId": message.get("requestId"),
            "error": f"Unknown message type: {message_type}",
        }

    def start(self, session_name: str) -> int:
        if self.session_id is not None:
            raise RuntimeError("A recording session is already active.")
        self.input_capture.reset()
        self.session_id = self.database.start_session(session_name)
        self.sequence_number = 0
        return self.session_id

    def stop(self) -> tuple[int, int]:
        if self.session_id is None:
            raise RuntimeError("No recording session is active.")
        session_id = self.session_id
        frame_count = self.sequence_number
        self.database.finish_session(session_id)
        self.session_id = None
        return session_id, frame_count

    def record_next(self) -> int | None:
        """Block for one socket message, reply, and record an ``ai_tick``."""
        if self.session_id is None:
            raise RuntimeError("Start a recording session before receiving ticks.")

        # This is the same receive pattern used by Env16.step().
        message = self.socket.recv_json()
        received_at_ns = time.time_ns()
        response = self.build_response(message)

        if not isinstance(message, dict) or message.get("type") != "ai_tick":
            self.socket.send_json(response)
            print(f"Ignored non-ai_tick payload: {message!r}", flush=True)
            return None

        world_state = message.get("worldState", {})
        player_input = self.input_capture.snapshot()

        # REP sockets must reply after every successful receive.
        self.socket.send_json(response)

        self.sequence_number += 1
        frame_id = self.database.save_frame(
            session_id=self.session_id,
            sequence_number=self.sequence_number,
            received_at_ns=received_at_ns,
            message=message,
            world_state=world_state,
            player_input=player_input,
            response=response,
        )
        print(
            f"\n[RECORDED] session={self.session_id} "
            f"frame={frame_id} sequence={self.sequence_number}",
            flush=True,
        )
        print(f"  action={json_text(player_input)}", flush=True)
        print(f"  state={json_text(world_state)}", flush=True)
        print(f"  response={json_text(response)}", flush=True)
        return frame_id

    def run(self, session_name: str) -> int:
        session_id = self.start(session_name)
        print(f"Recording session {session_id} ({session_name!r}).", flush=True)
        print("Waiting inside socket.recv_json() for ai_tick...", flush=True)
        print("Press Ctrl+C to stop recording.", flush=True)

        try:
            while True:
                self.record_next()
        except KeyboardInterrupt:
            print("\nStopping recording...", flush=True)
        finally:
            stopped_session, frame_count = self.stop()
            print(
                f"Session {stopped_session} saved {frame_count} frames.",
                flush=True,
            )
        return frame_count

    def close(self) -> None:
        if self.session_id is not None:
            self.stop()
        if self.owns_socket:
            self.socket.close(linger=0)
        self.database.close()


def parse_no_op_action(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record live state/action pairs to SQLite."
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
        type=parse_no_op_action,
        default="NoOp",
        help='JSON value returned as move (default: "NoOp")',
    )
    parser.add_argument(
        "--session-name",
        help="Start immediately with this session name instead of prompting",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Record empty inputs for manual action labeling",
    )
    return parser.parse_args(argv)


def default_session_name() -> str:
    return datetime.now().strftime("recording-%Y%m%d-%H%M%S")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_capture = NoInputCapture() if args.no_keyboard else WindowsInputCapture()
    recorder = PlayerRecorder(
        database_path=args.database,
        bind_url=args.bind_url,
        no_op_action=args.no_op_action,
        input_capture=input_capture,
    )

    print(f"Recorder bound to {args.bind_url}", flush=True)
    print(f"SQLite database: {recorder.database.path}", flush=True)
    try:
        session_name = args.session_name
        if not session_name:
            suggested_name = default_session_name()
            entered_name = input(
                f"Recording name [{suggested_name}] "
                "(press Enter to start): "
            ).strip()
            session_name = entered_name or suggested_name
        recorder.run(session_name)
    except KeyboardInterrupt:
        print("\nRecorder cancelled.", flush=True)
    finally:
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
