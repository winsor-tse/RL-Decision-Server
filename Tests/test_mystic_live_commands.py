from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from Automation.Bridge import ws_zmq_bridge as bridge
from Inference.Mystic_Live.commands import LiveCommands, COMMAND_SCHEMA
from Inference.Mystic_Live.inputs import LivePolicyInputs
from Inference.Mystic_Live.lifecycle import SingleRun
from Tests.test_mystic_live_lifecycle import request, Clock, StubActor


class CommandTests(TestCase):
    def setUp(self):
        self.commands = LiveCommands()
        self.world = request(0)['worldState']
        self.world['player'].update(id='42', mapX=50, mapY=50)
        self.world['entities'] = []

    def monster(self, identity, x, y, **kwargs):
        return dict(id=identity, type='monster', isCurrentPlayer=False,
                    mapX=x, mapY=y, hp=10, maxHp=10, mp=0, maxMp=0, **kwargs)

    def test_all_actions_and_empty_target_set(self):
        for action, expected in enumerate(('up', 'down', 'left', 'right', None,
                                            'castSpell:2:42', 'castSpell:3:42')):
            command = self.commands.encode(self.world, action)
            self.assertEqual(command['move'], expected)
            self.assertEqual(command['commandSchema'], COMMAND_SCHEMA)

    def test_euclidean_ranking_ties_rectangle_and_wire_identity(self):
        # Manhattan favors (6,0), Euclidean favors (4,4).
        self.world['entities'] = [self.monster(1, 56, 50), self.monster('002', 54, 54)]
        self.assertEqual(self.commands.encode(self.world, 4)['targetId'], '002')
        self.world['entities'] = [self.monster(10, 66, 60), self.monster('2', 34, 40),
                                  self.monster(0, 67, 50), self.monster(3, 50, 61)]
        self.assertEqual(self.commands.encode(self.world, 4)['move'], 'castSpell:1:2')
        self.world['entities'].reverse()
        self.assertEqual(self.commands.encode(self.world, 6)['targetId'], '2')
        self.world['entities'] = [self.monster(1, 51, 50)]
        self.world['entities'][0]['hp'] = 0
        self.world['entities'].append(dict(id=6, type='player', isCurrentPlayer=False))
        self.assertIsNone(self.commands.encode(self.world, 4)['move'])

    def test_self_namespace_is_separate_from_monster_id(self):
        self.world['entities'] = [self.monster('42', 51, 50)]
        self.assertEqual(self.commands.encode(self.world, 5)['targetKind'], 'monster')

    def test_capability_gate_and_no_target_preserves_chosen_action(self):
        clock = Clock()
        actor = StubActor(actions=(4, 5))
        inputs = LivePolicyInputs(71)
        run = SingleRun(actor, input_adapter=inputs, command_adapter=self.commands,
                        progression='snapshot', monotonic=clock)
        frame = request(0)
        frame['worldState'] = deepcopy(self.world)
        self.assertEqual(run.handle(frame)['idleReason'], 'extension_update_required')
        self.assertEqual(actor.starts, 0)
        frame['capabilities'] = [COMMAND_SCHEMA]
        frame['requestId'] = 'new'
        frame['worldState']['timestamp'] += 200
        result = run.handle(frame)
        self.assertIsNone(result['move'])
        self.assertEqual(result['actionLabel'], 'castSpell:1')
        next_frame = deepcopy(frame)
        next_frame['requestId'] = 'next'
        next_frame['worldState']['timestamp'] += 200
        result = run.handle(next_frame)
        self.assertEqual(result['move'], 'castSpell:2:42')
        self.assertEqual(actor.inputs[-1][0][60], 1)  # Previous chosen Arcane action.
        with patch.object(bridge, 'active_run_id', None), patch.object(bridge, 'terminal_reply', None):
            self.assertEqual(bridge.single_run_response(result, 'next')['targetKind'], 'self')
