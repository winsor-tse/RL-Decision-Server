"""Phase 2 movement engine; combat/effect events are explicit trace-only hooks."""
from numbers import Integral

from .movement import OFFSETS, SERVER_DIRECTIONS, legal_move, next_step_basic, outside_spawn_area
from .observation import direction_from_delta, euclidean_distance
from .scheduler import Scheduler
from .state import Direction


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
        self.scheduler = Scheduler(world)
        self.rng = TracedRng(rng, self.emit)
        self.pending_npc = {e.entity_id: e.sequence for e in world.events if e.kind == "npc_update"}

    def emit(self, kind, entity_id=None, **fields):
        if self.trace_enabled:
            self.trace.append(dict(time_ms=self.world.time_ms, kind=kind, entity_id=entity_id, **fields))

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
                delay = 50 if speed <= 500 else min(speed // 2, speed - 500)
                # Timer.Reset(negative) is modeled as a future timer origin.
                entity.next_attack_ms = self.world.time_ms + speed + delay
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
            self.emit("npc_attack", npc.entity_id, target=p.entity_id, damage_applied=False)
            npc.next_attack_ms = now + npc.stats.attack_ms
        self.queue_npc(npc)

    def schedule_respawn(self, npc_id):
        """Schedule a lifecycle notification for Phase 3's actual respawn handler."""
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
                self.emit("regeneration", p.entity_id, resources_applied=False)
                p.next_regen_ms = self.world.time_ms + self.config.timing.regen_ms
                self.scheduler.schedule(p.next_regen_ms, "regeneration", p.entity_id)
        elif event.kind == "respawn":
            npc = self.world.monsters.get(event.entity_id)
            if npc is not None and not npc.alive and npc.life_generation == event.generation:
                self.emit("respawn_due", npc.entity_id, respawn_applied=False)
        elif event.kind.startswith(("effect_tick:", "effect_expiry:")):
            kind, effect_id = event.kind.split(":", 1)
            for effect in list(self.world.effects):
                if (effect.effect_id, effect.target_id, effect.generation) != (effect_id, event.entity_id, event.generation):
                    continue
                if kind == "effect_expiry":
                    self.world.effects.remove(effect)
                else:
                    self.emit("effect_tick", event.entity_id, effect_id=effect_id, damage_applied=False)
                    effect.next_tick_ms += effect.interval_ms
                    # Tick at expiry is excluded; effects are [start, expiry).
                    if effect.next_tick_ms < effect.expires_at_ms:
                        self.scheduler.schedule(effect.next_tick_ms, event.kind, event.entity_id, event.generation)
        else:
            raise ValueError(f"Unsupported event kind {event.kind!r}")

    def advance(self, action):
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < 8:
            raise ValueError("Mystic action must be an integer from 0 through 7")
        self.trace = []
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
        else:
            applied = False
            reason = "player_dead" if not p.alive else "gear_disabled" if action == 4 else "spell_not_implemented"
        self.emit("action", p.entity_id, action=int(action), applied=applied, reason=reason)
        self.scheduler.advance(self.world.time_ms + self.config.timing.step_ms, self.dispatch)
        self.world.step_count += 1
        return applied, reason
