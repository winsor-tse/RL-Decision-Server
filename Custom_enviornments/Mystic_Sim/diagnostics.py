"""Detached Gym info payloads: callers cannot mutate engine state through info."""
from dataclasses import asdict
from copy import deepcopy
from .observation import nearest_monsters
from .targeting import select_target
from .rewards import collision_penalty


def build_info(env, *, action=None, applied=None, reason=None, components=None,
               outcome=None, end_reason=None):
    w, engine = env.world, env.engine
    target = select_target(w, next(s for s in env.config.spells if s.slot == 1)) if action is None else None
    target_id = (target.entity_id if target else None) if action is None else engine.selected_target_id
    return {
        "seed": int(env.np_random_seed), "profile": env.config.profile,
        "reward_profile": env.reward_config.profile,
        "simulation_time_ms": w.time_ms, "current_step": w.step_count,
        "kills": w.kills, "selected_target": target_id,
        "selected_entity_ids": [m.entity_id for m in nearest_monsters(w)],
        "action": None if action is None else int(action),
        "action_applied": applied, "action_failure_reason": reason,
        "reward_components": dict(components or {}),
        "collision_kind": w.player_collision_kind,
        "collision_streak": w.player_collision_streak,
        "collision_penalty": collision_penalty(w, env.reward_config),
        "cooldowns": asdict(w.player.cooldowns),
        "active_effects": [asdict(e) for e in w.effects],
        "damage_events": [asdict(e) for e in engine.damage_events],
        "death_events": [asdict(e) for e in engine.death_events],
        "respawn_events": [asdict(e) for e in engine.respawn_events],
        "cast_events": [asdict(e) for e in engine.cast_events],
        "combat_implemented": True, "spells_implemented": True, "reward_implemented": True,
        "episode_outcome": outcome, "episode_end_reason": end_reason, "is_win": outcome == "win",
        **({"trace": deepcopy(engine.trace)} if env.trace_enabled else {}),
    }
