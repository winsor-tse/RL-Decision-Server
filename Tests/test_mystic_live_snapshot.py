"""Snapshot progression uses fresh observations, never inferred acknowledgements."""
from unittest import TestCase
from unittest.mock import patch

from Automation.Bridge import ws_zmq_bridge as bridge
from Inference.Mystic_Live.inputs import LivePolicyInputs
from Inference.Mystic_Live.lifecycle import SingleRun
from Tests.test_mystic_live_lifecycle import Clock, StubActor, request


class SnapshotTests(TestCase):
    def make_run(self):
        self.clock = Clock()
        self.actor = StubActor()
        return SingleRun(self.actor, input_adapter=LivePolicyInputs(56),
                         progression='snapshot', monotonic=self.clock,
                         acknowledgement_timeout=.1)

    def test_repeated_decisions_keep_outcomes_unknown(self):
        run = self.make_run()
        with patch.object(run.history, 'apply', side_effect=AssertionError('confirmed history changed')):
            for index in range(4):
                frame = request(index, clock=self.clock)
                frame['execution'] = {'applied': True}  # Irrelevant in snapshot mode.
                reply = run.handle(frame)
                self.assertIsNotNone(reply['move'])
                self.assertIsNone(run.pending)
                self.clock.advance(.3)  # Longer than acknowledgement timeout.
                run.check_deadlines()
                self.assertFalse(run.terminal)
        view = run.snapshot()
        self.assertEqual(len(view['issued_commands']), 4)
        self.assertTrue(all(c['execution_outcome'] is None for c in view['issued_commands']))
        self.assertEqual(view['completed_actions'], [])
        self.assertIsNone(view['history']['action_applied'])
        self.assertEqual(view['timing']['confirmed_execution_intervals']['samples'], 0)
        self.assertEqual(view['timing']['client_snapshot_intervals']['samples'], 3)
        self.assertEqual(self.actor.starts, 1)

    def test_duplicate_stale_foreign_and_inspection_do_not_advance(self):
        run = self.make_run()
        first = request(0, clock=self.clock)
        run.handle(first)
        for frame in (first, request(0, connection='foreign'),
                      dict(request(1), worldState=first['worldState'])):
            self.assertIsNone(run.handle(frame)['move'])
        for _ in range(3):
            run.snapshot()
        self.assertEqual(run.memory, 1)
        self.assertEqual(run.action_count, 1)
        self.clock.advance()
        self.assertIsNotNone(run.handle(request(2, clock=self.clock))['move'])

    def test_stop_timeout_and_map_change_are_irreversible(self):
        for reason in ('manual_stop', 'state_timeout', 'unsupported_map', 'player_death'):
            with self.subTest(reason=reason):
                run = self.make_run()
                run.handle(request(0, clock=self.clock))
                frame = request(1, clock=self.clock)
                if reason == 'manual_stop':
                    run.stop()
                elif reason == 'state_timeout':
                    self.clock.advance(3)
                    run.check_deadlines()
                elif reason == 'unsupported_map':
                    frame['worldState']['player']['mapID'] = 'map54'
                else:
                    frame['worldState']['player']['hp'] = 0
                self.assertIsNone(run.handle(frame)['move'])
                self.assertEqual(run.stop_reason, reason)
                self.assertIsNone(run.handle(request(2))['move'])
                self.assertEqual(run.action_count, 1)

    def test_snapshot_rejects_outcome_inputs_and_default_stays_strict(self):
        with self.assertRaisesRegex(ValueError, 'outcome-independent'):
            SingleRun(input_adapter=LivePolicyInputs(80), progression='snapshot')
        self.assertEqual(SingleRun().progression, 'confirmed_feedback')

    def test_bridge_requires_explicit_snapshot_mode_for_running_commands(self):
        response = dict(requestId='one', runId='run', actionId='run:1', move='up',
                        reset=False, inferenceStatus='running')
        for mode, allowed in ((None, False), ('snapshot', True), ('unknown', False)):
            with patch.object(bridge, 'terminal_reply', None), patch.object(bridge, 'active_run_id', None):
                reply = dict(response)
                if mode is not None:
                    reply['progression'] = mode
                result = bridge.single_run_response(reply, 'one')
                self.assertEqual(result['move'], 'up' if allowed else None)
