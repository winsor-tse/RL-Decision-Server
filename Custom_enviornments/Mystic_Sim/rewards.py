"""Versioned simulator rewards; no live reward imports, file I/O, or ID-loss heuristics."""
COMPONENTS = ("health_state", "positioning", "damage_taken", "damage_dealt", "terminal", "killed")


def empty_components():
    return dict.fromkeys(COMPONENTS, 0.0)


def episode_status(world, config):
    if not world.player.alive:
        return True, False, "loss", "player_death"
    if world.kills >= config.win_kills:
        return True, False, "win", "kill_goal"
    if config.y_bounds is not None and not config.y_bounds[0] <= world.player.y <= config.y_bounds[1]:
        return False, True, "truncated", "y_boundary"
    if world.step_count >= config.max_episode_steps:
        return False, True, "truncated", "step_limit"
    return False, False, None, None


def calculate(world, config, previous_obs, obs, action, damage_events, death_events, outcome):
    parts = empty_components()
    player = world.player
    taken = sum(e.damage / player.max_hp for e in damage_events if e.target_id == player.entity_id)
    dealt = sum(e.damage / world.monsters[e.target_id].max_hp for e in damage_events
                if e.attacker_id == player.entity_id and e.target_id in world.monsters)
    killed = len({e.entity_id for e in death_events
                  if e.killer_id == player.entity_id and e.entity_id in world.monsters})
    if config.profile == "combat_reward_v1":
        # Attribute damage before regeneration; HP costs are not enemy damage.
        parts["health_state"] = -config.time_cost
        parts["damage_taken"] = -config.player_damage_weight * taken
        parts["damage_dealt"] = config.enemy_damage_weight * dealt
        parts["killed"] = config.kill_bonus * killed
        parts["terminal"] = -config.death_penalty if outcome == "loss" else 0.0
    else:
        # Snapshot of live Mystic/Env_conditions reward coefficients and branches.
        # Enemy damage and kills use engine events instead of disappearance guesses.
        hp = float(obs[3])
        parts["health_state"] = -.50 if hp < .25 else -.15 if hp < .50 else 0.0
        if action <= 3 and obs[0] == previous_obs[0] and obs[1] == previous_obs[1]:
            parts["positioning"] -= 10
        if config.legacy_y_penalty_below is not None and obs[1] < config.legacy_y_penalty_below:
            parts["positioning"] -= 100 * (config.legacy_y_penalty_below - float(obs[1]))
        closest = float(obs[6])
        parts["positioning"] += min(3, 9 - min(abs(closest - 5), 4) ** 2)
        hp_lost = float(previous_obs[3]) - hp
        # Intentionally preserved ONLY in legacy, including positive healing reward.
        if hp_lost > 0 or hp != .5:
            parts["damage_taken"] = -20 * hp_lost
        parts["damage_dealt"] = 25 * dealt
        parts["killed"] = 10.0 * killed
        parts["terminal"] = -100.0 if outcome == "loss" else 0.0
    return {key: float(value) for key, value in parts.items()}
