import json
import sqlite3
import tempfile
import unittest
import uuid
from contextlib import closing
from pathlib import Path

import zmq

from Offline.record_player import (
    RecorderController,
    RecorderServer,
    RecordingStore,
    virtual_key_name,
)


class FakeInputMonitor:
    def __init__(self):
        self.reset_count = 0
        self.snapshots = [
            {
                "capture_available": True,
                "keys_down": [{"key": "W", "vk": 87}],
                "events": [
                    {
                        "event": "key_down",
                        "key": "W",
                        "timestamp_unix_ns": 123,
                        "vk": 87,
                    }
                ],
            }
        ]

    def start(self):
        return None

    def stop(self):
        return None

    def reset(self):
        self.reset_count += 1

    def snapshot(self):
        if self.snapshots:
            return self.snapshots.pop(0)
        return {
            "capture_available": True,
            "keys_down": [],
            "events": [],
        }


class OfflineRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "recordings.db"
        self.store = RecordingStore(self.database_path)
        self.input_monitor = FakeInputMonitor()
        self.controller = RecorderController(
            self.store,
            self.input_monitor,
            no_op_action="NoOp",
        )

    def tearDown(self):
        self.store.close()
        self.temporary_directory.cleanup()

    def test_records_full_payload_and_input_only_during_session(self):
        message = {
            "type": "ai_tick",
            "requestId": "tick-7",
            "worldState": {"playerHp": 75, "nested": {"x": 12}},
            "extra": [1, 2, 3],
        }

        idle_response = self.controller.handle_message(message)
        self.assertEqual(idle_response["move"], "NoOp")

        session_id = self.controller.start_recording("combat example")
        response = self.controller.handle_message(message)
        stopped_session_id, frame_count = self.controller.stop_recording()

        self.assertEqual(stopped_session_id, session_id)
        self.assertEqual(frame_count, 1)
        self.assertEqual(response["type"], "ai_result")
        self.assertEqual(response["requestId"], "tick-7")
        self.assertEqual(response["move"], "NoOp")
        self.assertFalse(response["reset"])
        self.assertEqual(self.input_monitor.reset_count, 1)

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
        self.assertIsNone(frame["action_label"])

    def test_manual_annotation_updates_action_columns(self):
        self.controller.start_recording("manual actions")
        self.controller.handle_message(
            {"type": "ai_tick", "requestId": 1, "worldState": {}}
        )
        self.controller.stop_recording()

        frame_id = self.store.list_recent_frames(1)[0]["id"]
        self.store.annotate_frame(frame_id, "4")

        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT action_label, action_source FROM frames WHERE id = ?",
                (frame_id,),
            ).fetchone()
        self.assertEqual(row, ("4", "manual"))

    def test_invalid_message_gets_error_and_is_not_recorded(self):
        session_id = self.controller.start_recording("invalid input")
        response = self.controller.handle_message({"type": "ping"})
        self.controller.stop_recording()

        self.assertEqual(response["type"], "error")
        self.assertEqual(self.store.count_frames(session_id), 0)

    def test_rejects_overlapping_recording_sessions(self):
        self.controller.start_recording("first")
        with self.assertRaisesRegex(RuntimeError, "already recording"):
            self.controller.start_recording("second")
        self.controller.stop_recording()

    def test_virtual_key_names_are_stable_for_mapping(self):
        self.assertEqual(virtual_key_name(87), "W")
        self.assertEqual(virtual_key_name(0x70), "F1")
        self.assertEqual(virtual_key_name(0xA2), "CTRL_LEFT")

    def test_zmq_server_round_trip_uses_no_op(self):
        endpoint = f"inproc://offline-recorder-{uuid.uuid4()}"
        server = RecorderServer(endpoint, self.controller)
        server.start()
        context = zmq.Context.instance()
        client = context.socket(zmq.REQ)
        client.setsockopt(zmq.LINGER, 0)
        client.setsockopt(zmq.RCVTIMEO, 2000)
        client.connect(endpoint)
        try:
            self.controller.start_recording("socket test")
            client.send_json(
                {"type": "ai_tick", "requestId": "abc", "worldState": {}}
            )
            response = client.recv_json()
            self.controller.stop_recording()
        finally:
            client.close(linger=0)
            server.stop()

        self.assertEqual(response["requestId"], "abc")
        self.assertEqual(response["move"], "NoOp")
        self.assertEqual(len(self.store.list_recent_frames(10)), 1)


if __name__ == "__main__":
    unittest.main()
