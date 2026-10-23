"""Map seven policy choices to explicit live commands, without simulating outcomes."""
from Custom_enviornments.Mystic_Sim.actions import ACTIONS
from Custom_enviornments.Mystic_Sim.targeting import distance, in_casting_range
from .inputs import LiveObservationEncoder, entity_key

COMMAND_SCHEMA = 'mystic_explicit_target_v1'


class LiveCommands:
    required_capability = COMMAND_SCHEMA
    def __init__(self, encoder=None):
        self.encoder = encoder or LiveObservationEncoder()

    def encode(self, payload, action):
        if type(action) is not int or not 0 <= action < len(ACTIONS):
            raise ValueError('Expected a seven-action index')
        result = dict(commandSchema=COMMAND_SCHEMA, actionLabel=ACTIONS[action],
                      move=ACTIONS[action], targetKind=None, targetId=None)
        if action < 4:
            return result
        world = self.encoder.world(payload)
        # The ordinary payload exposes no magic-immunity flag. Only observable
        # eligibility is used here; the actual game remains the authority.
        candidates = [m for m in world.monsters.values() if in_casting_range(world.player, m)]
        if candidates:
            selected = min(candidates, key=lambda m: (distance(world.player, m), m.entity_id))
            target = next(e['id'] for e in payload['entities']
                          if e.get('type') == 'monster' and entity_key(e['id']) == selected.entity_id)
            kind = 'monster'
        elif action in (5, 6):
            target, kind = payload['player']['id'], 'self'
        else:
            return dict(result, move=None, idleReason='no_eligible_target')
        # A structured ID preserves the original namespace/type. The suffix is
        # retained for the existing parser but must be representable losslessly.
        if ':' in str(target):
            raise ValueError('Target ID cannot contain a command delimiter')
        return dict(result, move=f'{ACTIONS[action]}:{target}', targetId=target, targetKind=kind)
