"""Absolute-time movement, combat, casting, and effect transactions."""
from numbers import Integral

from .movement import OFFSETS, SERVER_DIRECTIONS, legal_move, next_step_basic, outside_spawn_area
from .observation import direction_from_delta, euclidean_distance
from .scheduler import Scheduler
from .state import Direction
from .state import PlayerState, DamageEvent, DeathEvent, RespawnEvent, CastEvent, TimedEffect
from . import combat, cooldowns, targeting, spells


class TracedRng:
    def __init__(self, generator, emit):
        self.generator, self.emit = generator, emit
        self.entity_id = None

    def roll(self, low, high, purpose):
        value = int(self.generator.integers(low, high + 1))
        self.emit("rng", self.entity_id, purpose=purpose, low=low, high=high, value=value)
        return value

    def chance(self, probability, purpose):
        return self.roll(0, 1000000000, purpose) <= probability * 1000000000


class Engine:
    def __init__(self, world, config, rng, *, trace=False):
        self.world, self.config = world, config
        self.trace_enabled = trace
        self.trace = []
        self.damage_events = []
        self.death_events = []
        self.respawn_events = []
        self.cast_events = []
        self.selected_target_id = None
        self.scheduler = Scheduler(world)
        self.rng = TracedRng(rng, self.emit)
        self.pending_npc = {e.entity_id: e.sequence for e in world.events if e.kind == "npc_update"}

    def emit(self, kind, entity_id=None, **fields):
        if self.trace_enabled:
            self.trace.append(dict(time_ms=self.world.time_ms, kind=kind, entity_id=entity_id, **fields))

    def entity(self, entity_id):
        return self.world.player if entity_id == self.world.player.entity_id else self.world.monsters.get(entity_id)

    def apply_damage(self, attacker, target, damage, *, raw_damage=None, damage_type="melee",
                     crit=False, dodged=False, blocked=False):
        if type(damage) is not int or damage < 0:
            raise ValueError("Damage must be a nonnegative integer")
        if target is None or not target.alive:
            return None
        before = target.hp
        target.hp = max(0, before-damage)
        event = DamageEvent(self.world.time_ms, attacker.entity_id, target.entity_id,
                            damage if raw_damage is None else raw_damage, before-target.hp,
                            before,target.hp,damage_type,crit,dodged,blocked)
        self.damage_events.append(event)
        self.emit("damage", target.entity_id, attacker=attacker.entity_id,
                  damage=event.damage, hp_before=before, hp_after=target.hp,
                  crit=crit, dodged=dodged, blocked=blocked)
        if not target.alive:
            self.handle_death(target, attacker.entity_id)
        elif not isinstance(target, PlayerState):
            self.add_damage_aggro(target.entity_id, attacker.entity_id)
        return event

    def handle_death(self, target, killer_id):
        if target.death_recorded:
            return
        target.death_recorded = True
        target.hp = 0
        if self.world.occupancy.get((target.x,target.y)) == target.entity_id:
            self.world.occupancy.pop((target.x,target.y))
        self.world.effects[:] = [e for e in self.world.effects
                                if target.entity_id not in (e.source_id,e.target_id)]
        for event in self.world.events:
            if event.entity_id == target.entity_id:
                self.scheduler.cancel(event.sequence)
        self.death_events.append(DeathEvent(self.world.time_ms,target.entity_id,killer_id))
        self.emit("death",target.entity_id,killer_id=killer_id)
        for npc in self.world.monsters.values():
            if npc.aggro_target == target.entity_id:
                npc.aggro_target = None
        if isinstance(target, PlayerState):
            target.cooldowns.slots.clear()
            target.cooldowns.families.clear()
            target.cooldowns.global_ready_at_ms = 0
        else:
            self.pending_npc.pop(target.entity_id,None)
            target.aggro_target = None
            target.life_generation += 1
            if killer_id == self.world.player.entity_id:
                self.world.kills += 1
            self.schedule_respawn(target.entity_id)

    def melee_attack(self, attacker, target):
        if not combat.can_hit(attacker,target):
            return None
        if isinstance(attacker,PlayerState) and not self.config.gear_enabled:
            raise ValueError("Player melee requires enabled gear")
        self.rng.entity_id = attacker.entity_id
        if self.rng.chance(combat.dodge_chance(getattr(target.stats,"dexterity",0)),"dodge"):
            return self.apply_damage(attacker,target,0,dodged=True)
        crit = False
        if isinstance(attacker,PlayerState):
            if not self.config.gear_enabled:
                raise ValueError("Player melee requires enabled gear")
            crit = self.rng.chance(combat.melee_crit_chance(attacker.stats.dexterity),"melee_crit")
            raw = int(self.config.gear.weapon_damage * combat.stat_factor(attacker.stats.strength)
                      * (combat.crit_multiplier(attacker.stats) if crit else 1))
        else:
            raw = attacker.stats.raw_damage  # baseline NPC Strength=0 gives factor 1
        damage,blocked = combat.mitigate(raw,attacker,target,self.rng)
        return self.apply_damage(attacker,target,damage,raw_damage=raw,crit=crit,blocked=blocked)

    def magic_damage(self, attacker, target, raw_damage, *, crit=False):
        if not combat.can_hit(attacker,target,magic=True):
            return None
        self.rng.entity_id = attacker.entity_id
        damage,blocked = combat.mitigate(raw_damage,attacker,target,self.rng,magic=True)
        return self.apply_damage(attacker,target,damage,raw_damage=raw_damage,
                                 damage_type="magic",crit=crit,blocked=blocked)

    def regenerate(self):
        p = self.world.player
        if not p.alive:
            return
        before=(p.hp,p.mp)
        if self.config.timing.regen_enabled:
            for resource,stat in (("hp",p.stats.stamina),("mp",p.stats.wisdom)):
                rate = getattr(p.stats,resource+"_regen_override")
                if rate is None:
                    rate = combat.regen_amount(stat,getattr(p,"max_"+resource),getattr(p.stats,"base_"+resource))
                amount = int(round(rate*self.config.timing.regen_ms/1000))
                setattr(p,resource,min(getattr(p,"max_"+resource),getattr(p,resource)+amount))
        self.emit("regeneration",p.entity_id,before=before,after=(p.hp,p.mp),
                  resources_applied=self.config.timing.regen_enabled)

    def cast(self, slot):
        p, now = self.world.player, self.world.time_ms
        spell = next(s for s in self.config.spells if s.slot == slot)
        target = targeting.select_target(self.world, spell)
        self.selected_target_id = target.entity_id if target else None
        reason = cooldowns.rejection(p, spell, now, target)
        if reason:
            self.emit("cast_rejected", p.entity_id, spell_id=spell.spell_id, reason=reason)
            return False, reason
        _, radius = spells.radii(spell, p.stats)
        partial = ()
        if slot == 1:
            targets = (target,)
        else:
            targets = targeting.area_targets(self.world, target, radius)
            if slot == 3:
                partial = targeting.area_targets(self.world, target, min(radius + .413, spell.max_radius), inner=radius)
        before_mp, before_hp = p.mp, p.hp
        p.mp -= spell.mp_cost
        p.hp = max(1, p.hp - spell.hp_cost)
        post_cost_mp = p.mp
        self.rng.entity_id = p.entity_id
        crit = self.rng.chance(combat.spell_crit_chance(p.stats.dexterity, p.stats.wisdom), "spell_crit")
        result = spells.calculate(spell, p.stats, p.mp, p.hp,
                                  tuple((m.entity_id, targeting.distance(target, m)) for m in targets + partial),
                                  caster_distance=targeting.distance(p, target),
                                  partial_ids=tuple(m.entity_id for m in partial), crit=crit,
                                  spell_multiplier=self.config.gear.spell_multiplier if self.config.gear_enabled else 0)

        def consume():
            p.mp -= result.mp_consumption
            p.hp -= result.hp_consumption
            self.emit("spell_resources", p.entity_id, spell_id=spell.spell_id,
                      mp_before=before_mp, mp_after_fixed=post_cost_mp, mp_after=p.mp,
                      hp_before=before_hp, hp_after=p.hp)

        for allocation in result.allocations:
            if slot == 2 and allocation.raw_damage <= 0:
                continue  # The Acid script skips its initial-damage loop at zero.
            victim = self.entity(allocation.target_id)
            # Acid's distance factor is applied after magic mitigation.
            damage, blocked = combat.mitigate(allocation.raw_damage, p, victim, self.rng, magic=True)
            damage = int(damage * allocation.falloff)
            if slot == 1:
                consume()  # Arcane spends percentage resources before main-target damage.
            self.apply_damage(p, victim, damage, raw_damage=allocation.raw_damage,
                              damage_type="magic", crit=crit, blocked=blocked)
        if slot != 1:
            consume()
        if slot == 2:
            for allocation in result.allocations:
                if self.entity(allocation.target_id).alive:
                    self.add_acid(spell, allocation, crit)
        cooldowns.start(p, spell, now, global_cooldown_ms=self.config.spell_rules.global_cooldown_ms)
        self.emit("cooldown_started", p.entity_id, slot=slot,
                  ready_at_ms=p.cooldowns.slots[slot], family=spell.family,
                  family_ready_at_ms=p.cooldowns.families.get(spell.family),
                  global_ready_at_ms=p.cooldowns.global_ready_at_ms)
        event = CastEvent(now, p.entity_id, spell.spell_id, target.entity_id,
                          tuple(a.target_id for a in result.allocations), before_mp, post_cost_mp,
                          p.mp, before_hp, p.hp, crit, p.cooldowns.slots[slot])
        self.cast_events.append(event)
        self.emit("cast", p.entity_id, spell_id=spell.spell_id, target_id=target.entity_id,
                  target_ids=event.target_ids, ready_at_ms=event.ready_at_ms,
                  radius=result.radius, raw_damage=result.total_damage, crit=crit)
        return True, None

    def add_acid(self, spell, allocation, crit, *, source_id=None):
        w = self.world
        source_id = w.player.entity_id if source_id is None else source_id
        generation = w.next_effect_sequence
        w.next_effect_sequence += 1
        effect_id = f"{spell.effect_id}:{source_id}"
        previous = [e for e in w.effects if (e.effect_id, e.source_id, e.target_id)
                    == (effect_id, source_id, allocation.target_id)]
        for old in previous:
            w.effects.remove(old)
            for event in w.events:
                if (event.entity_id == old.target_id and event.generation == old.generation
                        and event.kind in ("effect_tick:" + effect_id, "effect_expiry:" + effect_id)):
                    self.scheduler.cancel(event.sequence)
        # Refresh replaces the snapshot and restarts duration and tick interval.
        # Other casters have distinct IDs, so their effects remain independent.
        effect = TimedEffect(effect_id, source_id, allocation.target_id,
                             w.time_ms + spell.tick_interval_ms, w.time_ms + spell.duration_ms,
                             spell.tick_interval_ms, generation, allocation.tick_damage, crit)
        self.schedule_effect(effect)
        self.emit("effect_refreshed" if previous else "effect_added", effect.target_id,
                  effect_id=effect_id, generation=generation, tick_damage=effect.tick_damage,
                  next_tick_ms=effect.next_tick_ms, expires_at_ms=effect.expires_at_ms)

    def tick_effect(self, effect):
        target, source = self.entity(effect.target_id), self.entity(effect.source_id)
        applied = bool(effect.tick_damage and target and source and target.alive and source.alive)
        if applied:
            # TickEffect callback directly calls Damaged: no immunity, block, or crit reroll.
            self.apply_damage(source, target, effect.tick_damage,
                              damage_type="dot", crit=effect.crit)
        self.emit("effect_tick", effect.target_id, effect_id=effect.effect_id,
                  generation=effect.generation, damage_applied=applied)

    def respawn(self, npc):
        box = npc.spawn_box
        now = self.world.time_ms
        if not any(legal_move(self.world,x,y) for y in range(box.y,box.y+box.height)
                   for x in range(box.x,box.x+box.width)):
            boundary = (now//self.config.timing.step_ms+1)*self.config.timing.step_ms
            npc.respawn_at_ms = boundary
            self.scheduler.schedule(boundary,"respawn",npc.entity_id,npc.life_generation)
            self.emit("respawn_pending",npc.entity_id,reason="spawn_box_full",retry_at_ms=boundary)
            return
        self.rng.entity_id = npc.entity_id
        while True:
            x = self.rng.roll(box.x,box.x+box.width-1,"respawn_x")
            y = self.rng.roll(box.y,box.y+box.height-1,"respawn_y")
            if legal_move(self.world,x,y): break
        npc.x,npc.y = npc.spawn_x,npc.spawn_y = x,y
        npc.hp,npc.mp = npc.max_hp,npc.max_mp
        npc.move_interval_ms = self.rng.roll(npc.stats.move_ms[0],npc.stats.move_ms[1],"respawn_move_interval")
        npc.next_move_ms = now+npc.move_interval_ms
        npc.next_attack_ms = now+npc.stats.attack_ms
        npc.next_aggro_ms = now+npc.stats.aggro_check_ms
        npc.last_aggro_update_ms = now
        npc.facing = Direction.UP
        npc.aggro_target = npc.respawn_at_ms = None
        npc.magic_immune = npc.death_recorded = False
        npc.life_generation += 1
        self.world.occupancy[(x,y)] = npc.entity_id
        self.respawn_events.append(RespawnEvent(now,npc.entity_id,x,y,npc.life_generation,npc.move_interval_ms))
        self.emit("respawn",npc.entity_id,position=(x,y),movement_ms=npc.move_interval_ms)
        self.queue_npc(npc)

    def distance(self, npc):
        p = self.world.player
        return euclidean_distance(npc.x, npc.y, p.x, p.y)

    def queue_npc(self, npc, due=None):
        previous = self.pending_npc.pop(npc.entity_id, None)
        if previous is not None:
            self.scheduler.cancel(previous)
        if not npc.alive:
            return
        now = self.world.time_ms
        if due is None:
            deadlines = [npc.next_move_ms]
            if npc.aggro_target is None:
                deadlines.append(npc.next_aggro_ms)
            elif self.distance(npc) <= npc.stats.attack_radius:
                deadlines.append(npc.next_attack_ms)
            # Expired timers (e.g. stationary adjacent facing) are polled at the
            # next decision interval instead of endlessly re-enqueueing at now.
            due = min(d if d > now else now + self.config.timing.step_ms for d in deadlines)
        self.pending_npc[npc.entity_id] = self.scheduler.schedule(due, "npc_update", npc.entity_id, npc.life_generation)

    def add_damage_aggro(self, npc_id, attacker_id):
        """Phase 3 damage pipeline hook. No HP mutation occurs here."""
        npc = self.world.monsters[npc_id]
        p = self.world.player
        if npc.alive and p.alive and attacker_id == p.entity_id:
            self.acquire(npc, "damage")
            self.queue_npc(npc, self.world.time_ms)

    def acquire(self, npc, reason):
        if npc.aggro_target is None:
            npc.aggro_target = self.world.player.entity_id
            npc.next_attack_ms = self.world.time_ms + npc.stats.attack_ms
            self.emit("aggro_acquired", npc.entity_id, target=npc.aggro_target, reason=reason)

    def move(self, entity, direction, *, npc=False):
        before = (entity.x, entity.y)
        dx, dy = OFFSETS[direction]
        destination = (entity.x + dx, entity.y + dy)
        moved = legal_move(self.world, *destination)
        entity.facing = direction
        if moved:
            self.world.occupancy.pop(before, None)
            entity.x, entity.y = destination
            self.world.occupancy[destination] = entity.entity_id
            if npc and entity.aggro_target is not None:
                speed = entity.stats.attack_ms
                # Partial reset: this is the remaining wait, not an extra interval.
                # Round half milliseconds up so an integer clock never attacks early.
                delay = 50 if speed <= 500 else min((speed + 1) // 2, speed - 500)
                entity.next_attack_ms = self.world.time_ms + delay
        if npc:
            entity.next_move_ms = self.world.time_ms + entity.move_interval_ms
            self.emit("on_move", entity.entity_id, effects_applied=False)
        self.emit("movement", entity.entity_id, before=before, after=(entity.x, entity.y),
                  facing=int(direction), applied=moved, reason=None if moved else "blocked")
        return moved

    def npc_movement(self, npc):
        now = self.world.time_ms
        if npc.aggro_target is not None:
            p = self.world.player
            if self.distance(npc) <= 1:
                npc.facing = direction_from_delta(p.x - npc.x, p.y - npc.y)
                self.emit("facing", npc.entity_id, before=(npc.x, npc.y), after=(npc.x, npc.y),
                          facing=int(npc.facing))
                return  # FaceEntity does not reset MoveTimer.
            direction = next_step_basic(self.world, npc, p.x, p.y, self.rng)
        else:
            home_distance = euclidean_distance(npc.x, npc.y, npc.spawn_x, npc.spawn_y)
            idle_ms = now - npc.last_aggro_update_ms
            random_move = idle_ms < 30000 or home_distance < 15
            if not random_move and idle_ms >= 300000 and home_distance >= 20:
                before = (npc.x, npc.y)
                if legal_move(self.world, npc.spawn_x, npc.spawn_y):
                    self.world.occupancy.pop(before, None)
                    npc.x, npc.y = npc.spawn_x, npc.spawn_y
                    self.world.occupancy[(npc.x, npc.y)] = npc.entity_id
                # Occupancy is authoritative even for the server's return jump.
                npc.next_move_ms = now + npc.move_interval_ms
                self.emit("spawn_return_jump", npc.entity_id, before=before, after=(npc.x, npc.y))
                return
            if not random_move or outside_spawn_area(npc):
                if (npc.x, npc.y) == (npc.spawn_x, npc.spawn_y):
                    npc.next_move_ms = now + npc.move_interval_ms
                    return
                direction = next_step_basic(self.world, npc, npc.spawn_x, npc.spawn_y, self.rng)
            else:
                direction = SERVER_DIRECTIONS[self.rng.roll(0, 3, "idle_direction")]
        self.move(npc, direction, npc=True)

    def update_npc(self, npc):
        now, p = self.world.time_ms, self.world.player
        self.rng.entity_id = npc.entity_id
        if npc.aggro_target is not None:
            if (npc.aggro_target != p.entity_id or not p.alive
                    or self.distance(npc) > npc.stats.aggro_drop_distance):
                self.emit("aggro_dropped", npc.entity_id, target=npc.aggro_target)
                npc.aggro_target = None
            elif now - npc.last_aggro_update_ms >= 1000:
                npc.last_aggro_update_ms = now
        if npc.aggro_target is None and now >= npc.next_aggro_ms:
            if p.alive and self.distance(npc) <= npc.stats.aggro_radius:
                self.acquire(npc, "radius")
            npc.next_aggro_ms = now + npc.stats.aggro_check_ms
            self.emit("aggro_check", npc.entity_id, target=npc.aggro_target)
        if now >= npc.next_move_ms:
            self.npc_movement(npc)
        if (npc.aggro_target is not None and now >= npc.next_attack_ms
                and self.distance(npc) <= npc.stats.attack_radius):
            self.melee_attack(npc,p)
            self.emit("npc_attack", npc.entity_id, target=p.entity_id, damage_applied=True)
            npc.next_attack_ms = now + npc.stats.attack_ms
        self.queue_npc(npc)

    def schedule_respawn(self, npc_id):
        """Schedule the original entity 50 seconds after death."""
        npc = self.world.monsters[npc_id]
        if npc.alive:
            raise ValueError("Cannot schedule respawn of a living NPC")
        if npc.respawn_at_ms is not None:
            return
        npc.respawn_at_ms = self.world.time_ms + npc.stats.respawn_ms
        self.scheduler.schedule(npc.respawn_at_ms, "respawn", npc_id, npc.life_generation)

    def schedule_effect(self, effect):
        if (effect.interval_ms <= 0 or effect.next_tick_ms < self.world.time_ms
                or effect.expires_at_ms <= effect.next_tick_ms):
            raise ValueError("Invalid effect interval/deadlines")
        if any(e.effect_id == effect.effect_id and e.target_id == effect.target_id for e in self.world.effects):
            raise ValueError("Cancel/remove an existing effect before replacing it")
        self.world.effects.append(effect)
        self.scheduler.schedule(effect.next_tick_ms, "effect_tick:" + effect.effect_id,
                                effect.target_id, effect.generation)
        self.scheduler.schedule(effect.expires_at_ms, "effect_expiry:" + effect.effect_id,
                                effect.target_id, effect.generation)

    def dispatch(self, event):
        self.emit("event", event.entity_id, event_kind=event.kind, sequence=event.sequence)
        if event.kind == "npc_update":
            self.pending_npc.pop(event.entity_id, None)
            npc = self.world.monsters.get(event.entity_id)
            if npc is not None and npc.alive and npc.life_generation == event.generation:
                self.update_npc(npc)
        elif event.kind == "regeneration":
            p = self.world.player
            if p.alive:
                self.regenerate()
                p.next_regen_ms = self.world.time_ms + self.config.timing.regen_ms
                self.scheduler.schedule(p.next_regen_ms, "regeneration", p.entity_id)
        elif event.kind == "respawn":
            npc = self.world.monsters.get(event.entity_id)
            if npc is not None and not npc.alive and npc.life_generation == event.generation:
                self.respawn(npc)
        elif event.kind.startswith(("effect_tick:", "effect_expiry:")):
            kind, effect_id = event.kind.split(":", 1)
            for effect in list(self.world.effects):
                if (effect.effect_id, effect.target_id, effect.generation) != (effect_id, event.entity_id, event.generation):
                    continue
                if kind == "effect_expiry":
                    if effect in self.world.effects:
                        self.world.effects.remove(effect)
                    self.emit("effect_expired", effect.target_id, effect_id=effect.effect_id,
                              generation=effect.generation)
                else:
                    self.tick_effect(effect)
                    effect.next_tick_ms += effect.interval_ms
                    if effect in self.world.effects and effect.next_tick_ms < effect.expires_at_ms:
                        self.scheduler.schedule(effect.next_tick_ms, event.kind, event.entity_id, event.generation)
        else:
            raise ValueError(f"Unsupported event kind {event.kind!r}")

    def advance(self, action, duration_ms=None):
        if duration_ms is not None and (type(duration_ms) is not int or duration_ms != self.config.timing.step_ms):
            raise ValueError("Advance duration must equal the configured decision interval")
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < 8:
            raise ValueError("Mystic action must be an integer from 0 through 7")
        self.trace = []
        self.damage_events = []
        self.death_events = []
        self.respawn_events = []
        self.cast_events = []
        self.selected_target_id = None
        p = self.world.player
        if action < 4 and p.alive:
            applied = self.move(p, Direction(int(action)))
            reason = None if applied else "blocked"
            if applied:
                # NPCScript.OnEntityMoved checks acquisition on player movement.
                for npc in sorted(self.world.monsters.values(), key=lambda n: n.entity_id):
                    if npc.alive and npc.aggro_target is None and self.distance(npc) <= npc.stats.aggro_radius:
                        self.acquire(npc, "player_moved")
                        self.queue_npc(npc, self.world.time_ms)
        elif action == 4 and p.alive and self.config.gear_enabled:
            if self.world.time_ms < p.next_attack_ms:
                applied,reason=False,"cooldown"
            else:
                dx,dy=OFFSETS[p.facing]
                target=self.entity(self.world.occupancy.get((p.x+dx,p.y+dy)))
                self.selected_target_id = target.entity_id if target else None
                applied=combat.can_hit(p,target)
                reason=None if applied else "no_target"
                if applied:
                    self.melee_attack(p,target)
                    p.next_attack_ms=self.world.time_ms+self.config.gear.attack_ms
        elif action >= 5:
            applied, reason = self.cast(int(action) - 4)
        else:
            applied = False
            reason = "player_dead" if not p.alive else "gear_disabled"
        self.emit("action", p.entity_id, action=int(action), applied=applied, reason=reason)
        self.scheduler.advance(self.world.time_ms + self.config.timing.step_ms, self.dispatch)
        self.world.step_count += 1
        return applied, reason
