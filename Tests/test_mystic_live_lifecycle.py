"""Single-run/death-stop fixtures are synthetic; no game or trained policy is used."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import queue
import threading
import time
import unittest
from unittest.mock import patch, AsyncMock

import websockets
import zmq
import zmq.asyncio

from Automation.Bridge import ws_zmq_bridge as bridge
from Inference.Mystic_Live.lifecycle import (
    BRIDGE_SCHEMA, EXECUTION_SCHEMA, PolicyChoice, RunState, SingleRun,
)
from Inference.Mystic_Live.server import serve, decode_message

ROOT = Path(__file__).resolve().parents[1]
FRAME = json.loads((ROOT / 'Tests/fixtures/live/phase8a_capture.jsonl').read_text().splitlines()[2])['request']


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, amount=.2):
        self.now += amount


class StubActor:
    def __init__(self, actions=(4, 0, 5, 6)):
        self.actions = actions
        self.starts = 0
        self.inputs = []

    def initial_state(self):
        self.starts += 1
        return 0

    def choose(self, world, history, memory):
        self.inputs.append((deepcopy(world), deepcopy(history), memory))
        action = self.actions[(len(self.inputs) - 1) % len(self.actions)]
        return PolicyChoice(action, memory + 1, tuple(float(i == action) for i in range(7)))


def request(index, *, clock=None, connection='connection', bridge_id='bridge', hp=9463):
    message = deepcopy(FRAME)
    message['requestId'] = f'tick-{index}'
    message['worldState']['timestamp'] = 1000 + index * 200
    message['worldState']['player']['hp'] = hp
    message['bridgeContext'] = dict(schema=BRIDGE_SCHEMA, bridgeId=bridge_id, connectionId=connection)
    if clock is not None:
        message['bridgeContext']['receivedMonotonicNs'] = int(clock() * 1e9)
    return message


def acknowledge(message, pending_reply, *, attempted=0, observed=200, applied=True,
                blocked=False, cooldown=False, label='castSpell:1', **overrides):
    message['execution'] = dict(schema=EXECUTION_SCHEMA, runId=pending_reply['runId'],
                                actionId=pending_reply['actionId'], requestId=pending_reply['requestId'],
                                actionLabel=label, applied=applied, movementBlocked=blocked,
                                cooldownRejected=cooldown, clockId='synthetic-server-clock',
                                attemptedAtMs=attempted, stateTimeMs=observed, **overrides)
    return message


class LifecycleTests(unittest.TestCase):
    def make_run(self, actor=True, **kwargs):
        self.clock = Clock()
        self.actor = StubActor() if actor is True else actor
        self.run = SingleRun(self.actor, run_id='synthetic-run', monotonic=self.clock,
                             wall_time=lambda: 123., **kwargs)
        return self.run

    def test_saved_sequence_fixture_preserves_correlated_history_and_death_frame(self):
        fixture = json.loads((ROOT / 'Tests/fixtures/live/phase8b_sequence.json').read_text())
        run = self.make_run(StubActor(actions=tuple(fixture['actor_actions'])))
        final_death = None
        for tick in fixture['ticks']:
            self.clock.advance(tick['advance_seconds'])
            frame = request(tick['index'], hp=tick.get('hp', 9463))
            if 'execution' in tick:
                frame['execution'] = tick['execution']
            response = run.handle(frame)
            self.assertEqual(response['move'], tick['expected_move'])
            self.assertFalse(response['reset'])
            if 'expected_stop' in tick:
                self.assertEqual(response['stopReason'], tick['expected_stop'])
                if final_death is None:
                    final_death = run.snapshot()
                else:
                    self.assertEqual(run.snapshot(), final_death)
        self.assertEqual(run.action_count, 3)
        self.assertEqual(len(run.history.completed), 2)
        self.assertEqual(run.history.view()['cast_age_ms']['1'], 490)
        self.assertEqual(run.current_state['player']['hp'], 0)

    def test_first_valid_state_initializes_once_without_reset(self):
        run = self.make_run()
        self.assertIsNone(run.handle({'type': 'ai_tick', 'requestId': 'bad'})['move'])
        invalid = request(0)
        invalid['worldState'].pop('player')
        self.assertIsNone(run.handle(invalid)['move'])
        self.assertEqual(self.actor.starts, 0)
        response = run.handle(request(1, clock=self.clock))
        self.assertEqual(response['move'], 'castSpell:1')
        self.assertFalse(response['reset'])
        self.assertEqual(self.actor.starts, 1)
        self.assertEqual(run.state, RunState.ACTION_PENDING)
        self.assertEqual(run.previous_state, request(1)['worldState'])
        self.assertIsNone(run.history.applied)
        self.assertIsNone(run.history.view()['cast_age_ms']['1'])

    def test_pending_does_not_infer_success_from_mp_or_position(self):
        run = self.make_run()
        first = run.handle(request(0))
        for i in range(1, 5):
            self.clock.advance(.1)
            frame = request(i)
            frame['worldState']['player'].update(mp=100, mapX=54)
            reply = run.handle(frame)
            self.assertIsNone(reply['move'])
            self.assertEqual(reply['idleReason'], 'awaiting_execution_acknowledgement')
        self.assertEqual(len(self.actor.inputs), 1)
        self.assertEqual(run.pending.action_id, first['actionId'])
        self.assertIsNone(run.history.applied)
        self.assertIsNone(run.history.cast_times[1])

    def test_confirmed_history_uses_actual_elapsed_time_and_applies_once(self):
        run = self.make_run()
        first = run.handle(request(0))
        self.clock.advance(.175)
        second = run.handle(acknowledge(request(1), first, attempted=1000, observed=1175))
        self.assertEqual(second['move'], 'up')
        history_input = self.actor.inputs[-1][1]
        self.assertEqual(history_input['previous_action'], 4)
        self.assertEqual(history_input['cast_age_ms']['1'], 175)
        self.assertTrue(history_input['cast_seen']['1'])
        self.assertEqual(run.memory, 2)
        self.clock.advance(.1)
        repeated = acknowledge(request(2), first, attempted=1000, observed=1175)
        self.assertEqual(run.handle(repeated)['idleReason'], 'duplicate_acknowledgement')
        self.assertEqual(len(self.actor.inputs), 2)
        self.assertEqual(len(run.history.completed), 1)
        self.clock.advance(.215)
        third = run.handle(acknowledge(request(3), second, attempted=1200, observed=1490,
                                      applied=False, blocked=True, label='up'))
        self.assertEqual(third['move'], 'castSpell:2')
        self.assertTrue(run.history.blocked)
        self.assertFalse(run.history.applied)
        self.assertEqual(run.history.view()['cast_age_ms']['1'], 490)
        timing = run.snapshot()['timing']['confirmed_execution_intervals']
        self.assertEqual(timing['samples'], 2)
        self.assertAlmostEqual(timing['minimum_ms'], 175)
        self.assertAlmostEqual(timing['maximum_ms'], 315)

    def test_cooldown_rejection_does_not_reset_successful_cast_age(self):
        run = self.make_run(StubActor(actions=(4, 4)))
        first = run.handle(request(0))
        self.clock.advance()
        second = run.handle(acknowledge(request(1), first))
        self.clock.advance()
        run.handle(acknowledge(request(2), second, attempted=200, observed=450,
                               applied=False, cooldown=True))
        self.assertTrue(run.history.cooldown_rejected)
        self.assertEqual(run.history.view()['cast_age_ms']['1'], 450)

    def test_duplicate_stale_foreign_and_wrong_ack_do_not_advance_memory(self):
        run = self.make_run()
        first = run.handle(request(0))
        self.assertEqual(run.handle(request(0))['idleReason'], 'duplicate_request')
        stale = request(1)
        stale['worldState']['timestamp'] = 1000
        self.assertEqual(run.handle(stale)['idleReason'], 'stale_snapshot')
        self.assertEqual(run.handle(request(2, connection='other'))['idleReason'], 'foreign_connection')
        foreign = request(3)
        foreign['runId'] = 'old-run'
        self.assertEqual(run.handle(foreign)['idleReason'], 'foreign_run')
        wrong = acknowledge(request(4), first)
        wrong['execution']['runId'] = 'old-run'
        self.assertEqual(run.handle(wrong)['idleReason'], 'foreign_run_acknowledgement')
        wrong = acknowledge(request(5), first)
        wrong['execution']['requestId'] = 'wrong-request'
        self.assertEqual(run.handle(wrong)['idleReason'], 'unmatched_acknowledgement')
        self.assertEqual(run.memory, 1)
        self.assertEqual(self.actor.starts, 1)
        self.assertFalse(run.history.completed)

    def test_death_freezes_final_frame_and_later_alive_ticks_never_restart(self):
        run = self.make_run()
        run.handle(request(0))
        self.clock.advance()
        death = request(1, hp=0)
        death['worldState']['player'].update(mapID='map3471', mapX=-1, mapY=-1)
        reply = run.handle(death)
        self.assertTrue(reply['stopInference'])
        self.assertIsNone(reply['move'])
        self.assertFalse(reply['reset'])
        self.assertEqual(reply['stopReason'], 'player_death')
        final = run.snapshot()
        for i in range(2, 5):
            self.clock.advance()
            self.assertIsNone(run.handle(request(i, connection='reconnected'))['move'])
        run.stop('later_reason', interrupted=True)
        self.assertEqual(run.snapshot(), final)
        self.assertEqual(self.actor.starts, 1)
        self.assertEqual(len(self.actor.inputs), 1)
        self.assertEqual(final['current_state']['player']['hp'], 0)
        self.assertGreater(final['previous_state']['player']['hp'], 0)
        self.assertIsNone(run.history.applied)

    def test_first_dead_state_never_initializes_actor(self):
        run = self.make_run()
        self.assertEqual(run.handle(request(0, hp=0))['stopReason'], 'player_death')
        self.assertFalse(run.initialized)
        self.assertEqual(self.actor.starts, 0)

    def test_training_goal_y_boundary_and_step_limit_are_not_live_conditions(self):
        run = self.make_run(actor=None)
        for i in range(4):
            frame = request(i)
            frame['worldState'].update(kills=999, step_count=99999)
            frame['worldState']['player']['mapY'] = 5
            self.assertIsNone(run.handle(frame)['move'])
            self.clock.advance()
        self.assertEqual(run.state, RunState.RUNNING)
        self.assertFalse(run.terminal)

    def test_timeouts_manual_stop_disconnect_and_map_change_are_final(self):
        for trigger, reason in (('startup', 'startup_timeout'), ('ack', 'acknowledgement_timeout'),
                                ('idle', 'state_timeout'), ('manual', 'manual_stop'),
                                ('disconnect', 'client_disconnected'), ('map', 'unsupported_map')):
            with self.subTest(trigger=trigger):
                run = self.make_run(actor=None if trigger == 'idle' else True)
                if trigger != 'startup':
                    run.handle(request(0))
                if trigger in ('startup', 'idle', 'ack'):
                    self.clock.advance(31 if trigger == 'startup' else 3)
                    run.check_deadlines()
                elif trigger == 'manual':
                    run.stop()
                elif trigger == 'disconnect':
                    event = request(1)
                    event.update(type='bridge_event', event='client_disconnected')
                    run.handle(event)
                else:
                    frame = request(1)
                    frame['worldState']['player']['mapID'] = 'map3471'
                    run.handle(frame)
                self.assertTrue(run.terminal)
                self.assertEqual(run.stop_reason, reason)
                if trigger == 'map':
                    self.assertEqual(run.current_state['player']['mapID'], 'map3471')
                starts = self.actor.starts if self.actor else 0
                self.assertIsNone(run.handle(request(10))['move'])
                self.assertEqual(self.actor.starts if self.actor else 0, starts)

    def test_new_explicit_run_has_fresh_history_and_memory(self):
        old = self.make_run()
        old.handle(request(0))
        old.stop('player_death')
        actor = StubActor()
        new = SingleRun(actor, run_id='new-explicit-launch', monotonic=self.clock)
        reply = new.handle(request(10))
        self.assertEqual(actor.starts, 1)
        self.assertEqual(new.memory, 1)
        self.assertTrue(reply['actionId'].startswith('new-explicit-launch:'))
        self.assertFalse(any(new.history.view()['cast_seen'].values()))

    def test_invalid_ack_clock_and_actor_failures_are_inert(self):
        for mutation, expected in (({'applied': True, 'movementBlocked': True}, 'invalid_acknowledgement'),
                                   ({'applied': 1}, 'invalid_acknowledgement'),
                                   ({'attemptedAtMs': 500, 'stateTimeMs': 200}, 'invalid_execution_timing')):
            run = self.make_run()
            first = run.handle(request(0))
            frame = acknowledge(request(1), first)
            frame['execution'].update(mutation)
            self.assertEqual(run.handle(frame)['idleReason'], expected)
            self.assertEqual(run.memory, 1)
            self.assertIsNone(run.history.applied)
        run = self.make_run()
        first = run.handle(request(0))
        second = run.handle(acknowledge(request(1), first))
        changed = acknowledge(request(2), second, attempted=200, observed=400, label='up')
        changed['execution']['clockId'] = 'different-clock'
        self.assertEqual(run.handle(changed)['stopReason'], 'execution_clock_changed')
        actor = StubActor()
        actor.choose = lambda *_: PolicyChoice(7)
        run = self.make_run(actor)
        self.assertEqual(run.handle(request(0))['stopReason'], 'invalid_policy_action')

    def test_actor_cannot_mutate_captured_input_and_inspection_is_read_only(self):
        actor = StubActor()
        choose = actor.choose
        def mutating(world, history, memory):
            result = choose(world, history, memory)
            world['player']['hp'] = 0
            history['cast_seen']['1'] = True
            return result
        actor.choose = mutating
        run = self.make_run(actor)
        original = request(0)
        run.handle(original)
        view = run.snapshot()
        view['current_state']['player']['hp'] = 0
        self.assertEqual(run.current_state['player']['hp'], 9463)
        self.assertEqual(run.previous_state, original['worldState'])
        self.assertFalse(run.history.view()['cast_seen']['1'])
        self.assertEqual(run.memory, 1)

    def test_slow_or_failing_actor_never_dispatches_after_deadline(self):
        run = self.make_run()
        choose = self.actor.choose
        def slow(*args):
            self.clock.advance(3)
            return choose(*args)
        self.actor.choose = slow
        reply = run.handle(request(0))
        self.assertIsNone(reply['move'])
        self.assertEqual(reply['stopReason'], 'state_timeout')
        self.assertEqual(run.action_count, 0)
        run = self.make_run()
        def fail():
            raise RuntimeError('fixture init failed')
        self.actor.initial_state = fail
        self.assertEqual(run.handle(request(0))['stopReason'], 'actor_initialization_error')

    def test_malformed_ack_is_inert_and_stale_queue_ends_run(self):
        run = self.make_run()
        first = run.handle(request(0))
        invalid = acknowledge(request(1), first)
        invalid['execution']['actionId'] = []
        self.assertEqual(run.handle(invalid)['idleReason'], 'invalid_acknowledgement')
        self.assertEqual(run.memory, 1)
        self.assertFalse(run.history.completed)
        run = self.make_run()
        stale = request(0, clock=self.clock)
        stale['bridgeContext']['receivedMonotonicNs'] -= 3_000_000_000
        self.assertEqual(run.handle(stale)['stopReason'], 'stale_bridge_request')
        self.assertEqual(self.actor.starts, 0)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_live80_inputs_and_feedback_survive_websocket_zmq_transport(self):
        await self.exercise('inputs')

    async def test_bridge_latches_death_and_reconnect_cannot_restart_backend(self):
        await self.exercise('death')

    async def test_bridge_disconnect_stops_pending_run(self):
        await self.exercise('disconnect')

    async def exercise(self, stop_kind):
        context = zmq.Context()
        backend = context.socket(zmq.REP)
        port = backend.bind_to_random_port('tcp://127.0.0.1')
        from Inference.Mystic_Live.inputs import LivePolicyInputs, OBSERVATION_CONTEXT_SCHEMA
        inputs = LivePolicyInputs(80) if stop_kind == 'inputs' else None
        run = SingleRun(StubActor(), input_adapter=inputs, idle_timeout=5)
        errors = queue.Queue()
        stop = threading.Event()
        def worker():
            try:
                serve(run, backend, stop_event=stop, drain_seconds=.25, poll_ms=10)
            except Exception as error:
                errors.put(error)
        thread = threading.Thread(target=worker)
        thread.start()
        async_context = zmq.asyncio.Context()
        with (patch.object(bridge, 'single_run', True), patch.object(bridge, 'terminal_reply', None),
              patch.object(bridge, 'active_connection_id', None),
              patch.object(bridge, 'active_run_id', None),
              patch.object(bridge, 'zmq_context', async_context), patch.object(bridge, 'zmq_socket', None),
              patch.object(bridge, 'zmq_lock', asyncio.Lock()),
              patch.object(bridge, 'ZMQ_BACKEND_URL', f'tcp://127.0.0.1:{port}')):
            try:
                async with websockets.serve(bridge.handle_websocket, '127.0.0.1', 0) as server:
                    ws_port = server.sockets[0].getsockname()[1]
                    async with websockets.connect(f'ws://127.0.0.1:{ws_port}') as client:
                        first = request(0)
                        first.pop('bridgeContext')
                        if inputs is not None:
                            first['observationContext'] = dict(schema=OBSERVATION_CONTEXT_SCHEMA,
                                clockId='synthetic-server-clock', stateTimeMs=1000)
                        await client.send(json.dumps(first))
                        choice = json.loads(await client.recv())
                        self.assertEqual(choice['move'], 'castSpell:1')
                        pending = request(1)
                        pending.pop('bridgeContext')
                        await client.send(json.dumps(pending))
                        self.assertIsNone(json.loads(await client.recv())['move'])
                        if inputs is not None:
                            confirmed = acknowledge(request(2), choice, attempted=1000, observed=1175)
                            confirmed['observationContext'] = dict(schema=OBSERVATION_CONTEXT_SCHEMA,
                                clockId='synthetic-server-clock', stateTimeMs=1175)
                            await client.send(json.dumps(confirmed))
                            self.assertEqual(json.loads(await client.recv())['move'], 'up')
                            self.assertEqual(run.actor.inputs[-1][0].shape, (80,))
                            self.assertAlmostEqual(float(run.actor.inputs[-1][0][66]), .0175, places=7)
                            self.assertEqual(run.snapshot()['current_policy_input']['size'], 80)
                        if stop_kind in ('death', 'inputs'):
                            death = request(3 if inputs is not None else 2, hp=0)
                            await client.send(json.dumps(death))
                            reply = json.loads(await client.recv())
                            self.assertEqual(reply['stopReason'], 'player_death')
                            for i in range(3, 5):
                                await client.send(json.dumps(request(i)))
                                later = json.loads(await client.recv())
                                self.assertIsNone(later['move'])
                                self.assertFalse(later['reset'])
                    # The disconnect control event is serialized through the
                    # same REQ/REP stream; give it time to complete.
                    for _ in range(100):
                        if run.terminal:
                            break
                        await asyncio.sleep(.01)
                    self.assertTrue(run.terminal)
                    self.assertEqual(run.stop_reason, 'client_disconnected' if stop_kind == 'disconnect' else 'player_death')
                    final = run.snapshot()
                    async with websockets.connect(f'ws://127.0.0.1:{ws_port}') as client:
                        await client.send(json.dumps(request(6)))
                        self.assertIsNone(json.loads(await client.recv())['move'])
                    self.assertEqual(run.snapshot(), final)
                    self.assertEqual(run.action_count, 2 if inputs is not None else 1)
                    self.assertEqual(run.actor.starts, 1)
            finally:
                stop.set()
                await asyncio.to_thread(thread.join, 3)
                if bridge.zmq_socket is not None:
                    bridge.zmq_socket.close(linger=0)
                async_context.term()
                backend.close(linger=0)
                context.term()
        self.assertFalse(thread.is_alive())
        if not errors.empty():
            raise errors.get()

    async def test_backend_timeout_latches_inert_replies_without_retry(self):
        sender = AsyncMock(return_value={'requestId': 'tick-0', 'move': None, 'reset': False,
                                        'error': 'ZeroMQ backend timeout'})
        with (patch.object(bridge, 'single_run', True), patch.object(bridge, 'terminal_reply', None),
              patch.object(bridge, 'active_connection_id', None),
              patch.object(bridge, 'active_run_id', None),
              patch.object(bridge, 'send_to_zmq_backend', sender)):
            async with websockets.serve(bridge.handle_websocket, '127.0.0.1', 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f'ws://127.0.0.1:{port}') as client:
                    for i in range(3):
                        await client.send(json.dumps(request(i)))
                        response = json.loads(await client.recv())
                        self.assertIsNone(response['move'])
                        self.assertTrue(response['stopInference'])
                self.assertEqual(sender.await_count, 1)

    async def test_backend_run_changes_and_legacy_resets_are_suppressed(self):
        base = dict(type='ai_result', requestId='one', move='up', reset=False,
                    inferenceStatus='action_pending', runId='first-run', actionId='first-run:1')
        with patch.object(bridge, 'active_run_id', None), patch.object(bridge, 'terminal_reply', None):
            self.assertEqual(bridge.single_run_response(base, 'one')['move'], 'up')
            restarted = dict(base, runId='unexpected-restart', requestId='two')
            reply = bridge.single_run_response(restarted, 'two')
            self.assertIsNone(reply['move'])
            self.assertEqual(reply['stopReason'], 'backend_run_changed')
        with patch.object(bridge, 'active_run_id', None), patch.object(bridge, 'terminal_reply', None):
            reply = bridge.single_run_response(dict(requestId='one', move='direction:up', reset=True), 'one')
            self.assertIsNone(reply['move'])
            self.assertFalse(reply['reset'])
            self.assertEqual(reply['stopReason'], 'incompatible_backend')
        with patch.object(bridge, 'active_run_id', None), patch.object(bridge, 'terminal_reply', None):
            reply = bridge.single_run_response(base, 'another-request')
            self.assertIsNone(reply['move'])
            self.assertEqual(reply['stopReason'], 'backend_request_mismatch')

    async def test_json_decoder_rejects_nonfinite_numbers(self):
        for raw in (b'{', b'\xff', b'{"hp":NaN}'):
            with self.assertRaises((ValueError, UnicodeDecodeError)):
                decode_message(raw)


if __name__ == '__main__':
    unittest.main()
