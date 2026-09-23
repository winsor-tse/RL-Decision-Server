"""Read-only cast eligibility and absolute cooldown deadlines."""
from .targeting import in_casting_range


def rejection(player, spell, now, target):
    if not player.alive:
        return "player_dead"
    if target is None:
        return "no_target"
    if not in_casting_range(player, target):
        return "out_of_range"
    if target.magic_immune:
        return "magic_immune"
    if now < player.cooldowns.slots.get(spell.slot, 0):
        return "cooldown"
    if spell.family and now < player.cooldowns.families.get(spell.family, 0):
        return "family_cooldown"
    if player.mp < spell.mp_cost:
        return "insufficient_mp"
    if player.hp < spell.hp_cost:
        return "insufficient_hp"
    return None


def start(player, spell, now):
    player.cooldowns.slots[spell.slot] = now + spell.cooldown_ms
    if spell.family:
        player.cooldowns.families[spell.family] = now + spell.family_cooldown_ms
