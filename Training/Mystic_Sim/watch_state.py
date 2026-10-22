"""Detached display data for policy watch; rendering never runs inference.

TODO(live): one explicitly launched frozen-actor inference run, stopping on death
without a reset/episode loop. A separate telemetry producer supplies compatible
snapshots/decisions; rewards and critic outputs are optional. Keep unavailable
telemetry explicit and correlate each decision with its input state; launching
a local simulator does not mirror an authoritative live world. See WATCH_POLICY.md.
"""
from dataclasses import dataclass
import numpy as np

from Custom_enviornments.Mystic_Sim.observation import encode_observation, nearest_monsters
from Custom_enviornments.Mystic_Sim.targeting import select_target
from .history import HISTORY_FEATURES


ACTION_NAMES = ('Move up', 'Move down', 'Move left', 'Move right',
                'Arcane Blast', 'Acid Cloud', 'Tempest Inferno')
FEATURE_NAMES = ('Player X', 'Player Y', 'Facing', 'Player HP fraction',
                 'Player MP fraction', 'Map ID') + tuple(
    f'Enemy {i + 1} {field}' for i in range(10)
    for field in ('distance', 'direction', 'HP fraction', 'MP fraction')) + tuple(
    f'Enemy {i + 1} {axis}' for i in range(5) for axis in ('dx', 'dy')) + tuple(HISTORY_FEATURES)


@dataclass(frozen=True)
class WatchSnapshot:
    step: int
    time_ms: int
    player: dict
    enemies: tuple
    spells: tuple
    history: tuple
    observation: tuple
    normalized_observation: tuple
    alive_enemies: int
    respawning: int
    target_id: int | None
    kills: int
    kill_goal: int
    step_limit: int
    total_reward: float
    reward_components: tuple
    last_action: int | None
    action_applied: bool | None
    failure_reason: str | None
    last_outcome: str | None
    end_reason: str | None
    events: tuple


@dataclass(frozen=True)
class PolicyDecision:
    """Probabilities belong to input_snapshot, before the selected action."""
    input_snapshot: WatchSnapshot
    probabilities: tuple
    action: int

    @property
    def entropy_bits(self):
        return -sum(p * np.log2(p) for p in self.probabilities if p > 0)


def snapshot_from_session(session):
    """Sim producer only: copy data once per decision/reset, outside the renderer."""
    env, info = session.env, session.info
    w, p = env.world, env.world.player
    obs = encode_observation(w)
    policy_obs = session.history.encode(obs) if session.history is not None else obs
    scale = session.agent.observation_scale.detach().cpu().numpy()
    enemies = []
    for rank, monster in enumerate(nearest_monsters(w)):
        offset = 6 + rank * 4
        enemies.append(dict(rank=rank + 1, entity_id=monster.entity_id,
                            distance=float(obs[offset]), direction=monster.facing.name.lower(),
                            relative_direction=int(obs[offset + 1]),
                            hp=monster.hp, max_hp=monster.max_hp,
                            hp_fraction=float(obs[offset + 2]), mp_fraction=float(obs[offset + 3]),
                            dx=float(obs[46 + 2 * rank]) if rank < 5 else None,
                            dy=float(obs[47 + 2 * rank]) if rank < 5 else None,
                            aggro=monster.aggro_target is not None))
    spells = []
    for spell, name in zip(env.config.spells, ACTION_NAMES[4:]):
        ready = max(p.cooldowns.slots.get(spell.slot, 0),
                    p.cooldowns.families.get(spell.family, 0), p.cooldowns.global_ready_at_ms)
        spells.append(dict(name=name, slot=spell.slot, mp_cost=spell.mp_cost,
                           cooldown_seconds=max(0, ready - w.time_ms) / 1000,
                           mana_consumption=spell.mana_consumption))
    target = select_target(w, env.config.spells[0])
    return WatchSnapshot(
        step=w.step_count, time_ms=w.time_ms,
        player=dict(x=p.x, y=p.y, facing=p.facing.name.lower(), hp=p.hp, max_hp=p.max_hp,
                    mp=p.mp, max_mp=p.max_mp, map_id=w.map.map_id, level=p.stats.level),
        enemies=tuple(enemies), spells=tuple(spells), history=tuple(policy_obs[56:]),
        observation=tuple(float(x) for x in policy_obs),
        normalized_observation=tuple(float(x) for x in policy_obs / scale),
        alive_enemies=sum(m.alive for m in w.monsters.values()),
        respawning=sum(not m.alive for m in w.monsters.values()),
        target_id=target.entity_id if target else None, kills=w.kills,
        kill_goal=env.reward_config.win_kills, step_limit=env.reward_config.max_episode_steps,
        total_reward=session.total_reward,
        reward_components=tuple(info['reward_components'].items()),
        last_action=info['action'], action_applied=info['action_applied'],
        failure_reason=info['action_failure_reason'],
        last_outcome=info['episode_outcome'], end_reason=info['episode_end_reason'],
        events=tuple(session.log))
