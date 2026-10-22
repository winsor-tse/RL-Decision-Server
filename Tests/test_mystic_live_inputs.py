"""Saved-payload and simulator parity; synthetic evidence, no live game or weights."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Custom_enviornments.Mystic_Sim.actions import ACTIONS, action_label
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.observation import encode_observation
from Inference.Mystic_Live.inputs import (
    InputUnavailable, LiveObservationEncoder, LivePolicyInputs, OBSERVATION_CONTEXT_SCHEMA,
    audit_capture, entity_key, main,
)
from Inference.Mystic_Live.lifecycle import BRIDGE_SCHEMA, EXECUTION_SCHEMA, PolicyChoice, SingleRun
from Training.Mystic_Sim.core import Agent
from Training.Mystic_Sim.history import ObservableHistory, HISTORY_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'Tests/fixtures/live/phase8c_inputs.json').read_text())


def context(stamp=1000, clock='synthetic-server'):
    return dict(schema=OBSERVATION_CONTEXT_SCHEMA, clockId=clock, stateTimeMs=stamp)


def completion(index, action=4, *, attempted=1000, observed=1175, applied=True, blocked=False, cooldown=False):
    action_id, request_id = f'run:{index}', f'tick-{index}'
    return dict(action_id=action_id, request_id=request_id, action=action,
                acknowledgement=dict(schema=EXECUTION_SCHEMA, runId='run', actionId=action_id,
                                     requestId=request_id, actionLabel=ACTIONS[action], applied=applied,
                                     movementBlocked=blocked, cooldownRejected=cooldown,
                                     clockId='synthetic-server', attemptedAtMs=attempted, stateTimeMs=observed))


def message(index, *, stamp=None):
    result = dict(type='ai_tick', requestId=f'tick-{index}', worldState=deepcopy(FIXTURE['world']),
                  bridgeContext=dict(schema=BRIDGE_SCHEMA, bridgeId='bridge', connectionId='connection'))
    result['worldState']['timestamp'] += index * 200
    if stamp is not None:
        result['observationContext'] = context(stamp)
    return result


class ArrayActor:
    def __init__(self, action=4):
        self.action = action
        self.starts = 0
        self.inputs = []

    def initial_state(self):
        self.starts += 1
        return 0

    def choose(self, observation, history, memory):
        self.inputs.append(observation.copy())
        return PolicyChoice(self.action, memory + 1)


class LiveInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.encoder = LiveObservationEncoder()

    def inputs(self, size=80):
        return LivePolicyInputs(size, encoder=self.encoder)

    def test_saved_ten_slot_payload_matches_hand_calculated_features(self):
        world = deepcopy(FIXTURE['world'])
        result = self.encoder.encode(world)
        np.testing.assert_array_equal(result, np.array(FIXTURE['expected_56'], dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)
        info = self.encoder.inspect(world)
        self.assertEqual([key[1] for key in info['ranked_ids']], FIXTURE['ranked_numeric_ids'])
        self.assertEqual(info['visible_living_monsters'], 11)
        world['entities'].reverse()
        for entity in world['entities']:
            entity['distance'] = 999999  # Redundant client distance cannot override coordinates.
        np.testing.assert_array_equal(self.encoder.encode(world), result)

    def test_real_example_shape_is_explicitly_map_adjusted_not_mistaken_for_live53(self):
        original = json.loads((ROOT / 'Tests/fixtures/simulation/example_full_state.json').read_text())
        with self.assertRaisesRegex(ValueError, 'unsupported_map'):
            self.encoder.encode(original)
        constructed = deepcopy(original)
        constructed['player']['mapID'] = 53
        observed = self.encoder.encode(constructed)
        self.assertEqual(observed.shape, (56,))
        self.assertTrue(self.encoder.space.contains(observed))
        # Same numeric ID in player and monster namespaces is legitimate.
        self.assertIn((0, 7), self.encoder.world(constructed).monsters)

    def test_actual_simulator_worlds_have_identical_base_tensors(self):
        env = MysticSimEnv()
        self.addCleanup(env.close)
        for seed in (7, 42):
            env.reset(seed=seed)
            world = env.world
            player = world.player
            payload = dict(timestamp=0, player=dict(id=player.entity_id, mapID='map53', mapX=player.x,
                           mapY=player.y, direction=player.facing.name.lower(), hp=player.hp,
                           maxHp=player.max_hp, mp=player.mp, maxMp=player.max_mp), entities=[])
            for monster in reversed(list(world.monsters.values())):
                payload['entities'].append(dict(id=str(monster.entity_id), type='monster',
                    isCurrentPlayer=False, mapX=monster.x, mapY=monster.y, hp=monster.hp,
                    maxHp=monster.max_hp, mp=monster.mp, maxMp=monster.max_mp))
            np.testing.assert_array_equal(self.encoder.encode(payload), encode_observation(world))

    def test_empty_visible_set_pads_but_absent_fields_and_out_of_range_tiles_fail(self):
        world = deepcopy(FIXTURE['world'])
        world['entities'] = []
        self.assertFalse(self.encoder.encode(world)[6:].any())
        for mutate in (
            lambda w: w.pop('entities'),
            lambda w: w['entities'].append(dict(id=1)),
            lambda w: w['entities'].append(dict(id=1, type='monster', isCurrentPlayer=False, hp=1)),
            lambda w: w['player'].pop('mp'),
            lambda w: w['player'].update(mapX=100),
            lambda w: w['player'].update(mapY=5.5),
            lambda w: w['player'].update(mapX=True),
            lambda w: w['player'].update(direction=2),
            lambda w: w['player'].update(hp=float('nan')),
        ):
            invalid = deepcopy(world)
            mutate(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.encoder.encode(invalid)
        # Even an enemy outside the selected ten slots must not hide missing telemetry.
        invalid = deepcopy(FIXTURE['world'])
        invalid['entities'][0].pop('mapX')
        with self.assertRaisesRegex(ValueError, 'mapX'):
            self.encoder.encode(invalid)

    def test_strict_id_normalization_and_stable_opaque_ties(self):
        self.assertEqual(entity_key(10), entity_key('10'))
        self.assertLess(entity_key(2), entity_key('10'))
        self.assertLess(entity_key(10), entity_key('npc:10'))
        for value in (True, 2., None, '', ' x '):
            with self.subTest(value=value), self.assertRaises(ValueError):
                entity_key(value)
        world = deepcopy(FIXTURE['world'])
        monster = deepcopy(world['entities'][-3])
        monster.update(id=2)
        duplicate = deepcopy(monster)
        duplicate['id'] = '2'
        world['entities'] = [monster, duplicate]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.encoder.encode(world)
        monster['id'], duplicate['id'] = 'npc:z', 'npc:a'
        world['entities'] = [monster, duplicate]
        self.assertEqual(self.encoder.inspect(world)['ranked_ids'], [[1, 'npc:a'], [1, 'npc:z']])

    def test_bounds_fractions_direction_ties_and_normalization_match_policy(self):
        world = deepcopy(FIXTURE['world'])
        world['player'].update(mapX=99, mapY=99, hp=200, maxHp=100, mp=50, maxMp=0, direction='idle')
        monster = deepcopy(world['entities'][-3])
        monster.update(id=2, mapX=0, mapY=0)
        world['entities'] = [monster]
        inputs = self.inputs(56)
        encoded = inputs.encode(world)
        raw = np.array(encoded['values'], dtype=np.float32)
        self.assertEqual(raw[2], 4)
        np.testing.assert_array_equal(raw[3:5], [1, 0])
        np.testing.assert_array_equal(raw[6:8], [198, 0])  # Vertical wins diagonal tie.
        np.testing.assert_array_equal(raw[46:48], [-99, -99])
        np.testing.assert_array_equal(inputs.low[46:], [-99] * 10)
        agent = Agent(inputs.high)
        np.testing.assert_array_equal(agent.normalized(torch.tensor(raw)).detach().numpy(), encoded['normalized'])
        self.assertEqual(encoded['normalized'][46], -1)

    def test_saved_transitions_match_observable_history_exactly_including_unseen_ages(self):
        inputs = self.inputs()
        history = ObservableHistory(self.encoder.space.high, blocked_cells=self.encoder.map.blocked_cells)
        world = deepcopy(FIXTURE['world'])
        base = self.encoder.encode(world)
        stamp = 1000
        initial = inputs.encode(world, observation_context=context(stamp))
        np.testing.assert_array_equal(initial['values'], history.encode(base))
        for index, transition in enumerate(FIXTURE['transitions'], 1):
            previous_stamp = stamp
            stamp += transition['elapsed_ms']
            item = completion(index, transition['action'], attempted=previous_stamp, observed=stamp,
                              applied=transition['applied'], blocked=transition['blocked'], cooldown=transition['cooldown'])
            encoded = inputs.encode(world, observation_context=context(stamp), completion=item)
            history.update(base, transition['action'], applied=transition['applied'], blocked=transition['blocked'],
                           cooldown_rejected=transition['cooldown'], elapsed_ms=transition['elapsed_ms'])
            np.testing.assert_array_equal(encoded['values'], history.encode(base))
            self.assertEqual(encoded['schema'], HISTORY_SCHEMA)
        # Slot 3 has never cast, but its age channel still advances in training.
        self.assertGreater(inputs.history.features[12], 0)
        self.assertEqual(inputs.history.features[15], 0)

    def test_measured_execution_delay_and_rejection_preserve_cast_age(self):
        inputs = self.inputs()
        inputs.encode(FIXTURE['world'], observation_context=context(1000))
        first = completion(1, attempted=1050, observed=1200)
        inputs.encode(FIXTURE['world'], observation_context=context(1200), completion=first)
        self.assertAlmostEqual(float(inputs.history.features[10]), .015, places=7)
        rejected = completion(2, attempted=1250, observed=1500, applied=False, cooldown=True)
        inputs.encode(FIXTURE['world'], observation_context=context(1500), completion=rejected)
        self.assertAlmostEqual(float(inputs.history.features[10]), .045, places=7)
        capped = completion(3, 0, attempted=1500, observed=20000)
        encoded = inputs.encode(FIXTURE['world'], observation_context=context(20000), completion=capped)
        np.testing.assert_array_equal(encoded['values'][66:69], [1, 1, 1])

    def test_history_cannot_use_missing_feedback_wrong_snapshot_clock_or_duplicates(self):
        inputs = self.inputs()
        with self.assertRaisesRegex(ValueError, 'authoritative observationContext'):
            inputs.encode(FIXTURE['world'])
        inputs.encode(FIXTURE['world'], observation_context=context(1000))
        features = inputs.history.features.copy()
        with self.assertRaises(InputUnavailable):
            inputs.encode(FIXTURE['world'], observation_context=context(1100))
        for ctx, item in (
            (context(1175, 'new-clock'), completion(1)),
            (context(1200), completion(1)),
            (context(1175), completion(1, attempted=999)),
        ):
            with self.subTest(context=ctx), self.assertRaises(ValueError):
                inputs.encode(FIXTURE['world'], observation_context=ctx, completion=item)
            np.testing.assert_array_equal(features, inputs.history.features)
        inputs.encode(FIXTURE['world'], observation_context=context(1175), completion=completion(1))
        features = inputs.history.features.copy()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inputs.encode(FIXTURE['world'], observation_context=context(1175), completion=completion(1))
        np.testing.assert_array_equal(features, inputs.history.features)

    def test_shared_map_sensor_edges_and_entity_occupancy_independence(self):
        env = MysticSimEnv()
        self.addCleanup(env.close)
        for x, y in ((0, 0), (99, 99), (50, 45), (50, 50)):
            world = deepcopy(FIXTURE['world'])
            world['player'].update(mapX=x, mapY=y)
            expected = ObservableHistory.from_env(env).encode(self.encoder.encode(world))
            actual = self.inputs().encode(world, observation_context=context())
            np.testing.assert_array_equal(actual['values'], expected)
            world['entities'] = []
            empty = self.inputs().encode(world, observation_context=context())
            np.testing.assert_array_equal(actual['values'][72:], empty['values'][72:])
            if x == 0:
                np.testing.assert_array_equal(actual['values'][72:74], [0, 1])
                np.testing.assert_array_equal(actual['values'][76:78], [0, 1])

    def test_lifecycle_actor_gets_raw56_and_exact_seven_action_order(self):
        for index, label in enumerate(ACTIONS):
            actor = ArrayActor(index)
            run = SingleRun(actor, input_adapter=self.inputs(56), run_id='run')
            reply = run.handle(message(0))
            self.assertEqual(reply['move'], label)
            self.assertEqual(reply['move'], action_label(index))
            np.testing.assert_array_equal(actor.inputs[0], FIXTURE['expected_56'])
            self.assertEqual(actor.inputs[0].dtype, np.float32)
            self.assertEqual(run.snapshot()['last_decision']['policy_input']['values'], FIXTURE['expected_56'])

    def test_lifecycle80_starts_only_with_clock_and_advances_only_after_matched_feedback(self):
        actor = ArrayActor()
        run = SingleRun(actor, input_adapter=self.inputs(), run_id='run')
        self.assertIsNone(run.handle(message(0))['move'])
        self.assertEqual(actor.starts, 0)
        first = run.handle(message(1, stamp=1000))
        self.assertEqual(first['move'], 'castSpell:1')
        self.assertIsNone(run.handle(message(2, stamp=1100))['move'])
        self.assertIsNone(run.snapshot()['current_policy_input'])
        frame = message(3, stamp=1175)
        frame['execution'] = completion(1)['acknowledgement']
        frame['execution']['requestId'] = first['requestId']
        second = run.handle(frame)
        self.assertEqual(second['move'], 'castSpell:1')
        self.assertEqual(actor.starts, 1)
        self.assertEqual(len(actor.inputs), 2)
        self.assertEqual(actor.inputs[-1].shape, (80,))
        self.assertAlmostEqual(float(actor.inputs[-1][66]), .0175, places=7)
        self.assertEqual(run.snapshot()['current_policy_input']['schema'], HISTORY_SCHEMA)

    def test_lifecycle_interrupts_instead_of_mixing_feedback_with_newer_world(self):
        actor = ArrayActor()
        run = SingleRun(actor, input_adapter=self.inputs(), run_id='run')
        first = run.handle(message(0, stamp=1000))
        wrong = message(1, stamp=1200)
        wrong['execution'] = completion(1)['acknowledgement']
        wrong['execution']['requestId'] = first['requestId']
        reply = run.handle(wrong)
        self.assertTrue(reply['stopInference'])
        self.assertEqual(reply['stopReason'], 'invalid_policy_input')
        self.assertEqual(len(actor.inputs), 1)
        self.assertEqual(run.action_count, 1)
        self.assertIn('same clock and snapshot', run.snapshot()['policy_input_status'])

    def test_observe_only_does_not_fill_later80_history_with_zeros(self):
        run = SingleRun(input_adapter=self.inputs())
        run.handle(message(0, stamp=1000))
        self.assertIsNotNone(run.snapshot()['current_policy_input'])
        run.handle(message(1, stamp=1200))
        self.assertEqual(run.snapshot()['policy_input_status'], 'awaiting_confirmed_transition')
        self.assertIsNone(run.snapshot()['current_policy_input'])
        self.assertIsNotNone(run.snapshot()['current_base_observation'])

    def test_slow_input_preparation_never_initializes_or_calls_a_stale_actor(self):
        actor = ArrayActor()
        inputs = self.inputs(56)
        now = [100.]
        run = SingleRun(actor, input_adapter=inputs, monotonic=lambda: now[0])
        encode = inputs.encode
        def slow(*args, **kwargs):
            result = encode(*args, **kwargs)
            now[0] += 3
            return result
        with patch.object(inputs, 'encode', side_effect=slow):
            reply = run.handle(message(0))
        self.assertTrue(reply['stopInference'])
        self.assertEqual(reply['stopReason'], 'state_timeout')
        self.assertEqual(actor.starts, 0)
        self.assertEqual(actor.inputs, [])

    def test_capture_audit_preserves_synthetic_origin_and_reports_missing_coverage(self):
        path = ROOT / 'Tests/fixtures/live/phase8a_capture.jsonl'
        report = audit_capture(path, encoder=self.encoder)
        self.assertGreater(report['valid_56'], 0)
        self.assertEqual(report['evidence_origins'], ['synthetic_scripted_fixture'])
        self.assertEqual(report['maximum_visible_living_monsters'], 1)
        self.assertFalse(report['ten_slot_coverage_observed'])
        self.assertEqual(report['authoritative_context_frames'], 0)
        self.assertTrue(any(error['error'] == 'unsupported_map' for error in report['errors']))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'audit.json'
            self.assertEqual(main(['--capture', str(path), '--output', str(output)]), 0)
            saved = json.loads(output.read_text())
            self.assertFalse(saved['live_parity_validated'])
            self.assertIsNone(saved['deployment_checkpoint'])
            with self.assertRaises(FileExistsError):
                main(['--capture', str(path), '--output', str(output)])


if __name__ == '__main__':
    unittest.main()
