"""Optional bounded JSONL evidence capture. No policy, game control or simulator I/O."""
from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path
import queue
import threading
import time
import uuid

SCHEMA = 'mystic_zmq_capture_v1'
LOGGER = logging.getLogger(__name__)


class JsonlCapture:
    """Copy on enqueue; disk writes happen on one worker, outside the bridge loop.

    Files are exclusive-create, never appended/overwritten. Overflow and limits
    drop evidence, never gameplay messages; capture_end reports all losses.
    Bridge observations alone do not establish command execution or game identity.
    """
    def __init__(self, path, *, label=None, action_labels=(), max_records=10000,
                 max_bytes=64 * 1024 * 1024, queue_size=256):
        if min(max_records, max_bytes, queue_size) <= 0:
            raise ValueError('Capture limits must be positive')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open('x', encoding='utf-8', newline='\n')
        self.session_id = uuid.uuid4().hex
        self.label, self.action_labels = label, tuple(action_labels)
        self.max_records, self.max_bytes = max_records, max_bytes
        self.sequence = self.written = self.bytes_written = 0
        self.queue_dropped = self.limit_dropped = 0
        self.error = None
        self.closed = False
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._write_loop, name='zmq-evidence', daemon=True)
        self._thread.start()
        self.record('capture_start', limits={'max_records': max_records, 'max_bytes': max_bytes},
                    action_labels=list(self.action_labels))

    def _row(self, event, fields):
        self.sequence += 1
        return dict(schema=SCHEMA, session_id=self.session_id, episode_id=None,
                    sequence=self.sequence, event=event,
                    wall_time_unix_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                    evidence_origin='bridge_observed_unverified_client', label=self.label, **fields)

    def record(self, event, **fields):
        if self.closed:
            return
        row = self._row(event, deepcopy(fields))
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.queue_dropped += 1

    def command(self, response):
        """Only the label/index in the configured contract is known at this boundary."""
        move = response.get('move') if isinstance(response, dict) else None
        label, target = move, None
        if isinstance(move, str) and move.startswith('castSpell:'):
            pieces = move.split(':')
            label = ':'.join(pieces[:2])
            target = pieces[2] if len(pieces) == 3 and pieces[2] else None
        return dict(action_label=label,
                    action_index=self.action_labels.index(label) if label in self.action_labels else None,
                    explicit_target_id=target, execution_confirmed=None,
                    action_applied_at_server_ms=None)

    def _write(self, row):
        line = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
        self._file.write(line)
        self._file.flush()
        self.bytes_written += len(line.encode('utf-8'))

    def _write_loop(self):
        try:
            while True:
                row = self._queue.get()
                try:
                    if row is None:
                        break
                    if self.error is not None:
                        continue
                    size = len((json.dumps(row, ensure_ascii=False, allow_nan=False,
                                           separators=(',', ':')) + '\n').encode('utf-8'))
                    if self.written >= self.max_records or self.bytes_written + size > self.max_bytes:
                        self.limit_dropped += 1
                        continue
                    self._write(row)
                    self.written += 1
                except (OSError, ValueError, TypeError) as error:
                    self.error = f'{type(error).__name__}: {error}'
                    LOGGER.error('Evidence capture failed; bridge traffic continues: %s', self.error)
                finally:
                    self._queue.task_done()
            if self.error is None:
                # Small completion marker is outside the evidence budget so a
                # full file can still explain its missing/truncated records.
                self._write(self._row('capture_end', self.status()))
        except OSError as error:
            self.error = f'{type(error).__name__}: {error}'
            LOGGER.error('Evidence capture could not finalize: %s', self.error)
        finally:
            self._file.close()

    def status(self):
        return dict(records_written=self.written, bytes_written=self.bytes_written,
                    queue_dropped=self.queue_dropped, limit_dropped=self.limit_dropped,
                    writer_error=self.error)

    def close(self):
        if not self.closed:
            self.closed = True
            self._queue.put(None)
            self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
