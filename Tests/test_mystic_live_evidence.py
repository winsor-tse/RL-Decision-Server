"""Phase 8A freezes evidence without choosing a model or changing game traffic."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import websockets
import zmq
import zmq.asyncio

from Automation.Bridge.evidence import JsonlCapture, SCHEMA
from Automation.Bridge import ws_zmq_bridge as bridge
from Inference.Mystic_Live.baseline import (
    ROOT, SCENARIOS, build_baseline, capture_summary, field_matrix, file_identity,
    markdown_report, read_capture, write_baseline,
)

FIXTURE = ROOT / 'Tests/fixtures/live/phase8a_capture.jsonl'


class EvidenceTests(unittest.TestCase):
    def test_capture_preserves_values_and_requires_explicit_action_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'capture.jsonl'
            request = dict(type='ai_tick', requestId='one', worldState={'player': {'hp': 100}})
            with JsonlCapture(path, action_labels=('up', 'castSpell:1')) as capture:
                capture.record('client_request', request=request)
                request['worldState']['player']['hp'] = 0
                command = capture.command({'move': 'castSpell:1:42'})
                self.assertEqual(command['action_index'], 1)
                self.assertEqual(command['explicit_target_id'], '42')
                self.assertIsNone(command['execution_confirmed'])
                self.assertIsNone(capture.command({'move': 'castSpell:2'})['action_index'])
                self.assertIsNone(capture.command({'move': 'castSpell:1'})['explicit_target_id'])
            rows = read_capture(path)
            self.assertEqual(rows[1]['request']['worldState']['player']['hp'], 100)
            self.assertEqual(rows[-1]['event'], 'capture_end')
            self.assertEqual(rows[-1]['queue_dropped'], 0)
            self.assertEqual(rows[-1]['limit_dropped'], 0)
            with self.assertRaises(FileExistsError):
                JsonlCapture(path)

    def test_limits_and_queue_loss_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bounded.jsonl'
            with JsonlCapture(path, max_records=1, queue_size=2) as capture:
                capture._queue.join()
                for _ in range(5):
                    capture.record('client_request', request={'data': 'x' * 1000})
            rows = read_capture(path)
            self.assertLessEqual(rows[-1]['records_written'], 1)
            self.assertEqual(rows[-1]['queue_dropped'] + rows[-1]['limit_dropped'], 5)
            self.assertFalse(capture_summary(path, rows)['clean_completion'])

    def test_writer_failure_is_reported_without_raising_into_bridge(self):
        with tempfile.TemporaryDirectory() as folder:
            capture = JsonlCapture(Path(folder) / 'failed.jsonl')
            capture._queue.join()
            with patch.object(capture, '_write', side_effect=OSError('disk full')):
                capture.record('test')
                capture.close()
            self.assertIn('disk full', capture.status()['writer_error'])

    def test_baseline_stays_unselected_and_reports_source_provenance(self):
        manifest = build_baseline(captures=[FIXTURE])
        self.assertEqual(manifest['checkpoint']['status'], 'unselected')
        self.assertIsNone(manifest['checkpoint']['metadata'])
        self.assertEqual(manifest['validation']['real_game_session'], 'not_verified')
        self.assertFalse(manifest['validation']['policy_live_compatible'])
        self.assertTrue(manifest['map']['hashes_match'])
        self.assertEqual(manifest['map']['blocked_cells'], 5088)
        self.assertEqual(manifest['payload_reference']['example_map_id'], 'map3471')
        summary = manifest['captures'][0]
        self.assertEqual(summary['origins'], ['synthetic_scripted_fixture'])
        self.assertEqual(summary['ai_tick_count'], 9)
        self.assertTrue(summary['clean_completion'])
        self.assertFalse(summary['unmatched_requests'])
        self.assertTrue(all(summary['scenario_coverage'][name] == 'labeled_only_execution_unverified'
                            for name in SCENARIOS))
        unavailable = {r['field'] for r in manifest['field_availability'] if r['evidence'] == 'unavailable'}
        self.assertIn('execution.action_applied', unavailable)
        self.assertIn('execution.successful_cast_time', unavailable)
        self.assertIn('identity.loaded_extension', unavailable)
        self.assertIn('identity.game_build', unavailable)
        self.assertIn('not verified', markdown_report(manifest))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'baseline'
            write_baseline(target, manifest)
            self.assertEqual(json.loads((target / 'baseline.json').read_text())['schema'], 'mystic_live_baseline_v1')
            with self.assertRaises(FileExistsError):
                write_baseline(target, manifest)

    def test_bad_and_partial_captures_do_not_become_verified(self):
        rows = read_capture(FIXTURE)
        changed = deepcopy(rows)
        changed[4]['sequence'] += 10
        changed.append(dict(rows[2], sequence=23, request_id='unanswered'))
        summary = capture_summary(FIXTURE, changed)
        self.assertTrue(summary['sequence_gaps'])
        self.assertTrue(summary['unmatched_requests'])
        self.assertFalse(summary['clean_completion'])
        empty = field_matrix([{'worldState': {'entities': 12}}], 'malformed_fixture')
        self.assertFalse(next(r for r in empty if r['field'] == 'worldState.entities')['complete'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'truncated.jsonl'
            path.write_text('{"schema":', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'invalid/truncated'):
                read_capture(path)

    def test_local_plugin_inventory_is_not_runtime_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            extension = Path(folder)
            (extension / 'manifest.json').write_text('{"name":"Fixture plugin","version":"1.2.3"}')
            (extension / 'battler.js').write_text('// fixture')
            manifest = build_baseline(extension_dir=extension)
            self.assertEqual(manifest['extension']['version'], '1.2.3')
            self.assertFalse(manifest['extension']['loaded_extension_verified'])
            self.assertEqual(manifest['extension']['files']['battler.js']['sha256'],
                             file_identity(extension / 'battler.js')['sha256'])

    def test_explicit_checkpoint_is_inventoried_without_running_or_restoring_it(self):
        import torch
        from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
        from Training.Mystic_Sim.core import Agent, PPOConfig, metadata
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.pt'
            env = MysticSimEnv()
            try:
                saved_metadata = metadata(env)
                model = Agent(env.observation_space.high)
            finally:
                env.close()
            torch.save(dict(format='mystic_sim_ppo_v2', metadata=saved_metadata,
                            ppo_config=asdict(PPOConfig()), agent=model.state_dict(), optimizer={},
                            update=0, global_step=0, rng={}, episodes=0, wins=0), path)
            rng = torch.get_rng_state().clone()
            with (patch.object(MysticSimEnv, 'reset', side_effect=AssertionError('baseline reset a world')),
                  patch.object(torch.optim, 'Adam', side_effect=AssertionError('baseline restored optimizer'))):
                baseline = build_baseline(checkpoint=path)
            self.assertEqual(baseline['checkpoint']['local_compatibility'], 'passed')
            self.assertEqual(baseline['checkpoint']['metadata'], saved_metadata)
            self.assertEqual(baseline['checkpoint']['sha256'], file_identity(path)['sha256'])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            # A changed schema is retained as incompatible evidence, not silently
            # relabeled as the current contract.
            payload = torch.load(path, weights_only=True)
            payload['metadata']['contract']['actions'][0] = 'attack'
            torch.save(payload, path)
            self.assertEqual(build_baseline(checkpoint=path)['checkpoint']['local_compatibility'], 'failed')


class BridgeEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_error_still_finalizes_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'startup.jsonl'
            with patch.object(bridge, 'reset_zmq_socket', side_effect=RuntimeError('fixture startup failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture startup failure'):
                    await bridge.main(['--capture', str(path)])
            rows = read_capture(path)
            self.assertEqual(rows[0]['event'], 'capture_start')
            self.assertEqual(rows[-1]['event'], 'capture_end')
            self.assertIsNone(bridge.capture)

    async def test_real_local_websocket_zmq_exchange_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            context = zmq.asyncio.Context()
            backend = context.socket(zmq.REP)
            port = backend.bind_to_random_port('tcp://127.0.0.1')
            capture_path = Path(folder) / 'transport.jsonl'
            capture = JsonlCapture(capture_path, action_labels=('up', 'down', 'left', 'right',
                                                               'castSpell:1', 'castSpell:2', 'castSpell:3'))
            requests = [r['request'] for r in read_capture(FIXTURE) if r['event'] == 'client_request']
            expected_responses = [r['response'] for r in read_capture(FIXTURE) if r['event'] == 'client_response_sent']

            async def reply():
                for expected, response in zip(requests, expected_responses):
                    incoming = await backend.recv_json()
                    self.assertEqual(incoming, expected)
                    await backend.send_json(response)

            with (patch.object(bridge, 'zmq_context', context), patch.object(bridge, 'zmq_socket', None),
                  patch.object(bridge, 'ZMQ_BACKEND_URL', f'tcp://127.0.0.1:{port}'),
                  patch.object(bridge, 'zmq_lock', asyncio.Lock()), patch.object(bridge, 'capture', capture)):
                task = asyncio.create_task(reply())
                try:
                    async with websockets.serve(bridge.handle_websocket, '127.0.0.1', 0) as server:
                        ws_port = server.sockets[0].getsockname()[1]
                        async with websockets.connect(f'ws://127.0.0.1:{ws_port}') as client:
                            # Both malformed JSON and a valid non-object JSON
                            # are recorded and leave the next exchange usable.
                            for raw in ('{', '[]'):
                                await client.send(raw)
                                self.assertIn('error', json.loads(await client.recv()))
                            for request, response in zip(requests, expected_responses):
                                await client.send(json.dumps(request))
                                self.assertEqual(json.loads(await client.recv()), response)
                        await asyncio.wait_for(task, timeout=5)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    if bridge.zmq_socket is not None:
                        bridge.zmq_socket.close(linger=0)
                    backend.close(linger=0)
                    context.term()
                    capture.close()
            rows = read_capture(capture_path)
            self.assertEqual(sum(r['event'] == 'backend_sent' for r in rows), 9)
            self.assertEqual(sum(r['event'] == 'malformed_message' for r in rows), 2)
            self.assertTrue(any(r['event'] == 'client_disconnected' for r in rows))
            summary = capture_summary(capture_path, rows)
            self.assertTrue(summary['clean_completion'])
            self.assertFalse(summary['request_id_mismatches'])
            self.assertEqual([r['request'] for r in rows if r['event'] == 'backend_sent'], requests)

    async def test_timeout_capture_and_disabled_capture_preserve_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            capture = JsonlCapture(Path(folder) / 'timeout.jsonl')
            socket = AsyncMock()
            socket.recv_json.side_effect = asyncio.TimeoutError
            with (patch.object(bridge, 'zmq_socket', socket), patch.object(bridge, 'capture', capture),
                  patch.object(bridge, 'zmq_lock', asyncio.Lock()), patch.object(bridge, 'reset_zmq_socket')):
                response = await bridge.send_to_zmq_backend({'requestId': 'timeout'})
                self.assertEqual(response['error'], 'ZeroMQ backend timeout')
            capture.close()
            self.assertIn('backend_timeout', [r['event'] for r in read_capture(capture.path)])
            socket = AsyncMock()
            socket.recv_json.return_value = {'requestId': 'plain', 'move': 'up', 'reset': False}
            with (patch.object(bridge, 'zmq_socket', socket), patch.object(bridge, 'capture', None),
                  patch.object(bridge, 'zmq_lock', asyncio.Lock())):
                response = await bridge.send_to_zmq_backend({'requestId': 'plain'})
                self.assertEqual(response, socket.recv_json.return_value)


if __name__ == '__main__':
    unittest.main()
