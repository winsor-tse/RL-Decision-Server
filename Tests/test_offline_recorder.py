import json
import socket
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path

import zmq

from Offline.record_player import NoInputCapture, PlayerRecorder, virtual_key_name


class FakeInputCapture:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def snapshot(self):
        return {
            "capture_available": True,
            "keys_down": [{"key": "W", "vk": 87}],
            "events": [{"event": "key_down", "key": "W", "vk": 87}],
        }


class FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.responses = []

    def recv_json(self):
        return self.messages.pop(0)

    def send_json(self, response):
        self.responses.append(response)


class OfflineRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "recordings.db"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_recorder(self, message, input_capture=None):
        fake_socket = FakeSocket([message])
        recorder = PlayerRecorder(
            database_path=self.database_path,
            no_op_action="NoOp",
            input_capture=input_capture or FakeInputCapture(),
            socket=fake_socket,
        )
        return recorder, fake_socket

    def test_record_next_matches_environment_receive_reply_pattern(self):
        message = {
            "type": "ai_tick",
            "requestId": "tick-7",
            "worldState": {"playerHp": 75, "nested": {"x": 12}},
            "extra": [1, 2, 3],
        }
        input_capture = FakeInputCapture()
        recorder, fake_socket = self.make_recorder(message, input_capture)
        try:
            session_id = recorder.start("combat example")
            frame_id = recorder.record_next()
            stopped_session, frame_count = recorder.stop()
        finally:
            recorder.close()

        self.assertEqual(stopped_session, session_id)
        self.assertEqual(frame_count, 1)
        self.assertEqual(frame_id, 1)
        self.assertEqual(input_capture.reset_count, 1)
        self.assertEqual(fake_socket.responses[0]["move"], "NoOp")
        self.assertEqual(fake_socket.responses[0]["requestId"], "tick-7")

        with closing(sqlite3.connect(self.database_path)) as connection:
            cursor = connection.execute("SELECT * FROM frames")
            row = cursor.fetchone()
            columns = [description[0] for description in cursor.description]
            frame = dict(zip(columns, row))

        self.assertEqual(json.loads(frame["payload_json"]), message)
        self.assertEqual(
            json.loads(frame["world_state_json"]),
            message["worldState"],
        )
        self.assertEqual(
            json.loads(frame["input_json"])["keys_down"],
            [{"key": "W", "vk": 87}],
        )
        self.assertEqual(json.loads(frame["response_json"])["move"], "NoOp")

    def test_non_ai_tick_gets_error_and_is_not_recorded(self):
        recorder, fake_socket = self.make_recorder({"type": "ping"})
        try:
            recorder.start("invalid")
            frame_id = recorder.record_next()
            recorder.stop()
        finally:
            recorder.close()

        self.assertIsNone(frame_id)
        self.assertEqual(fake_socket.responses[0]["type"], "error")
        with closing(sqlite3.connect(self.database_path)) as connection:
            frame_count = connection.execute(
                "SELECT COUNT(*) FROM frames"
            ).fetchone()[0]
        self.assertEqual(frame_count, 0)

    def test_no_input_capture_supports_manual_labeling(self):
        capture = NoInputCapture()
        self.assertEqual(capture.snapshot()["keys_down"], [])
        self.assertFalse(capture.snapshot()["capture_available"])

    def test_rejects_overlapping_sessions(self):
        recorder, _ = self.make_recorder(
            {"type": "ai_tick", "worldState": {}}
        )
        try:
            recorder.start("first")
            with self.assertRaisesRegex(RuntimeError, "already active"):
                recorder.start("second")
            recorder.stop()
        finally:
            recorder.close()

    def test_virtual_key_names_are_stable_for_mapping(self):
        self.assertEqual(virtual_key_name(87), "W")
        self.assertEqual(virtual_key_name(0x61), "NUMPAD1")
        self.assertEqual(virtual_key_name(0x67), "NUMPAD7")
        self.assertEqual(virtual_key_name(0x70), "F1")
        self.assertEqual(virtual_key_name(0xA2), "CTRL_LEFT")

    def test_real_tcp_zmq_round_trip_records_and_returns_no_op(self):
        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            port = port_probe.getsockname()[1]
        endpoint = f"tcp://127.0.0.1:{port}"
        recorder = PlayerRecorder(
            database_path=self.database_path,
            bind_url=endpoint,
            no_op_action="NoOp",
            input_capture=NoInputCapture(),
        )
        recorder.start("tcp")
        response_holder = {}

        def send_tick():
            context = zmq.Context.instance()
            client = context.socket(zmq.REQ)
            client.setsockopt(zmq.LINGER, 0)
            client.setsockopt(zmq.RCVTIMEO, 2000)
            client.connect(endpoint)
            try:
                client.send_json(
                    {
                        "type": "ai_tick",
                        "requestId": "tcp-check",
                        "worldState": {"hp": 99},
                    }
                )
                response_holder["response"] = client.recv_json()
            finally:
                client.close(linger=0)

        client_thread = threading.Thread(target=send_tick)
        client_thread.start()
        try:
            recorder.record_next()
            client_thread.join(timeout=2)
            _, frame_count = recorder.stop()
        finally:
            recorder.close()

        self.assertFalse(client_thread.is_alive())
        self.assertEqual(response_holder["response"]["move"], "NoOp")
        self.assertEqual(frame_count, 1)
        with closing(sqlite3.connect(self.database_path)) as connection:
            saved_frames = connection.execute(
                "SELECT COUNT(*) FROM frames"
            ).fetchone()[0]
        self.assertEqual(saved_frames, 1)


if __name__ == "__main__":
    unittest.main()
