"""Spell numeric regressions and isolated/full-world cast transactions."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from numpy.testing import assert_array_equal

from Custom_enviornments.Mystic_Sim import spells, targeting, cooldowns
from Custom_enviornments.Mystic_Sim.config import (
    PlayerConfig, ScenarioConfig, SpellRules, TimingConfig, RewardConfig, baseline_spells,
)
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.engine import Engine
from Tests.test_mystic_sim_combat import ScriptedRng


class SpellTests(unittest.TestCase):
    def scene(self, positions=((51, 50),), config=None, rolls=None):
        env = MysticSimEnv(config=config, trace=True)
        env.reset(seed=42)
        w = env.world
        w.player.x, w.player.y = 50, 50
        w.monsters = {i: w.monsters[i] for i in range(2, 2 + len(positions))}
        w.occupancy = {(50, 50): 1}
        for monster, position in zip(w.monsters.values(), positions):
            monster.x, monster.y = position
            w.occupancy[position] = monster.entity_id
        w.events.clear()
        env.engine = Engine(w, env.config, env.np_random, trace=True)
        # Isolate spell transactions from NPC pursuit; the replay test uses real NPCs.
        env.engine.queue_npc = lambda *args, **kwargs: None
        if rolls is not None:
            env.engine.rng = ScriptedRng(rolls)
        self.addCleanup(env.close)
        return env, w, env.engine

    def assert_rejected_unchanged(self, w, engine, slot, reason):
        before = deepcopy(w)
        calls = list(engine.rng.calls)
        events = (list(engine.damage_events), list(engine.cast_events))
        self.assertEqual(engine.cast(slot), (False, reason))
        self.assertEqual(w, before)
        self.assertEqual(engine.rng.calls, calls)
        self.assertEqual((engine.damage_events, engine.cast_events), events)

    def test_numeric_goldens_and_pure_calculation(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/simulation/spells_expected.json').read_text())
        stats = PlayerConfig()
        for case in fixture['cases']:
            with self.subTest(slot=case['slot'], crit=case['crit']):
                spell = baseline_spells()[case['slot'] - 1]
                result = spells.calculate(spell, stats, 15108-spell.mp_cost, 9463,
                                          ((2, 0),), caster_distance=1, crit=case['crit'])
                self.assertEqual((result.base_damage, result.total_damage), (case['base'], case['total']))
                self.assertEqual(result.allocations[0].raw_damage, case['initial'])
                self.assertEqual(result.allocations[0].tick_damage, case['tick'])
                self.assertEqual(result.mp_consumption, case['charge'])
                self.assertEqual(stats, PlayerConfig())
                env, w, engine = self.scene(rolls=[case['crit'], False])
                self.assertEqual(engine.cast(spell.slot), (True, None))
                self.assertEqual(w.player.mp, case['mp_after'])
                event = engine.damage_events[0]
                self.assertEqual(event.raw_damage, case['initial'])
                self.assertEqual(event.crit, case['crit'])
                self.assertEqual(w.monsters[2].hp, w.monsters[2].max_hp-event.damage)
                self.assertEqual(w.monsters[2].aggro_target, 1)
                self.assertEqual([p for p, _ in engine.rng.calls], ['spell_crit', 'block'])
                self.assertEqual(w.player.cooldowns.slots[spell.slot], spell.cooldown_ms)
                self.assertEqual(engine.cast_events[0].mp_after_fixed, 15108-spell.mp_cost)
                self.assertEqual(len(w.effects), int(spell.slot == 2))

    def test_nearest_euclidean_tie_break_dead_and_immune(self):
        env, w, engine = self.scene(((53, 53), (55, 50), (47, 47)), rolls=[])
        self.assertEqual(targeting.select_target(w, env.config.spells[0]).entity_id, 2)
        w.monsters[2].magic_immune = True
        self.assertEqual(targeting.select_target(w, env.config.spells[0]).entity_id, 4)
        w.monsters[4].hp = 0
        self.assertEqual(targeting.select_target(w, env.config.spells[0]).entity_id, 3)
        w.monsters[3].magic_immune = True
        self.assert_rejected_unchanged(w, engine, 1, 'no_target')

    def test_no_target_self_fallback_and_no_hit_cost(self):
        for slot, expected in ((1,15108),(2,9452),(3,14358)):
            env, w, engine = self.scene((), rolls=[] if slot == 1 else [False])
            if slot == 1:
                self.assert_rejected_unchanged(w, engine, slot, 'no_target')
            else:
                self.assertEqual(engine.cast(slot), (True, None))
                self.assertEqual(engine.cast_events[0].target_id, 1)
                self.assertEqual(engine.cast_events[0].target_ids, ())
                self.assertEqual([p for p, _ in engine.rng.calls], ['spell_crit'])
                self.assertEqual(w.effects, [])
            self.assertEqual(w.player.mp, expected)
            self.assertEqual(engine.damage_events, [])

    def test_rejections_no_resources_state_or_rng(self):
        for spell in baseline_spells():
            for reason in ('insufficient_mp', 'insufficient_hp', 'cooldown', 'player_dead'):
                configured = replace(spell, hp_cost=10)
                config = ScenarioConfig(spells=tuple(configured if s.slot == spell.slot else s for s in baseline_spells()))
                env, w, engine = self.scene(config=config, rolls=[])
                if reason == 'insufficient_mp': w.player.mp = spell.mp_cost-1
                if reason == 'insufficient_hp': w.player.hp = 9
                if reason == 'cooldown': w.player.cooldowns.slots[spell.slot] = 1
                if reason == 'player_dead': w.player.hp = 0
                self.assert_rejected_unchanged(w, engine, spell.slot, reason)
        env, w, engine = self.scene(rolls=[])
        w.player.cooldowns.families['arcane-blast'] = 1
        self.assert_rejected_unchanged(w, engine, 1, 'family_cooldown')

    def test_exact_affordability_hp_clamp_and_resource_rounding(self):
        for spell in baseline_spells():
            env, w, engine = self.scene(rolls=[False,False])
            w.player.mp = spell.mp_cost
            self.assertTrue(engine.cast(spell.slot)[0])
            self.assertEqual(w.player.mp, 0)
        # 2.5 rounds to 2; 7.5 rounds to 8 (midpoint-to-even).
        for mp, expected in ((10,2),(30,8)):
            spell = replace(baseline_spells()[0], mana_consumption=.25, vita_consumption=.5)
            result = spells.calculate(spell, PlayerConfig(), mp, 5, ((2,0),))
            self.assertEqual(result.mp_consumption, expected)
            self.assertEqual(result.hp_consumption, 2)
        configured = replace(baseline_spells()[0], hp_cost=9463)
        env,w,engine = self.scene(config=ScenarioConfig(spells=(configured, *baseline_spells()[1:])), rolls=[False,False])
        self.assertTrue(engine.cast(1)[0])
        self.assertEqual(w.player.hp, 1)

    def test_resource_and_damage_source_order(self):
        for slot in (1,2,3):
            env,w,engine = self.scene(rolls=[False,False])
            before = deepcopy(w)
            spell = env.config.spells[slot-1]
            spells.calculate(spell,w.player.stats,w.player.mp-spell.mp_cost,w.player.hp,((2,0),))
            self.assertEqual(w,before)
            self.assertEqual(engine.rng.calls,[])
            engine.cast(slot)
            kinds = [e['kind'] for e in engine.trace]
            if slot == 1:
                self.assertLess(kinds.index('spell_resources'), kinds.index('damage'))
            else:
                self.assertLess(kinds.index('damage'), kinds.index('spell_resources'))

    def test_cooldown_millisecond_and_decision_boundaries(self):
        for slot, ready, boundary in ((1,3500,3600),(2,5000,5000),(3,1750,1800)):
            env,w,engine = self.scene((), rolls=[False]*10) if slot == 2 else self.scene(rolls=[False]*10)
            engine.cast(slot)
            w.time_ms = ready-1
            self.assert_rejected_unchanged(w,engine,slot,'cooldown')
            w.time_ms = ready
            w.player.mp = w.player.max_mp
            self.assertTrue(engine.cast(slot)[0])
            env,w,engine = self.scene((), rolls=[False]*10) if slot == 2 else self.scene(rolls=[False]*10)
            self.assertTrue(env.step(slot+4)[4]['action_applied'])
            while w.time_ms < boundary:
                self.assertFalse(env.step(slot+4)[4]['action_applied'])
            w.player.mp = w.player.max_mp
            self.assertTrue(env.step(slot+4)[4]['action_applied'])
            self.assertEqual(engine.cast_events[0].time_ms,boundary)

    def test_arcane_is_single_target_and_block_mitigates(self):
        damage = []
        for blocked in (False, True):
            env,w,engine = self.scene(((51,50),(51,51)), rolls=[False,blocked])
            engine.cast(1)
            self.assertEqual([e.target_id for e in engine.damage_events],[2])
            self.assertEqual(w.monsters[3].hp,w.monsters[3].max_hp)
            damage.append(engine.damage_events[0].damage)
        self.assertLess(damage[1],damage[0])

    def test_acid_exact_area_edge_falloff_and_shared_crit(self):
        # Override cap to a lattice-representable distance for inclusive edge tests.
        acid = replace(baseline_spells()[1], max_radius=4.0)
        cfg = ScenarioConfig(spells=(baseline_spells()[0],acid,baseline_spells()[2]))
        env,w,engine = self.scene(((51,50),(55,50),(55,51),(52,50)),config=cfg,rolls=[True,False,False,False])
        self.assertEqual([m.entity_id for m in targeting.area_targets(w,w.monsters[2],4)], [2,3,5])
        engine.cast(2)
        self.assertEqual([e.target_id for e in engine.damage_events],[2,3,5])
        self.assertTrue(all(e.crit for e in engine.damage_events))
        self.assertLess(engine.damage_events[1].damage,engine.damage_events[0].damage)
        self.assertEqual(len({e.tick_damage for e in w.effects}),1)
        self.assertEqual([p for p,_ in engine.rng.calls],['spell_crit','block','block','block'])
        raw, radius = spells.radii(baseline_spells()[1],PlayerConfig())
        self.assertAlmostEqual(raw,7.261143448765107)
        self.assertEqual(radius,4.25)

    def test_aoe_immunity_filters_before_allocation(self):
        for slot in (2,3):
            env,w,engine = self.scene(((51,50),(51,51)),rolls=[False,False])
            w.monsters[3].magic_immune = True
            engine.cast(slot)
            self.assertEqual([e.target_id for e in engine.damage_events],[2])
            self.assertEqual(w.monsters[3].hp,w.monsters[3].max_hp)
        env,w,engine = self.scene((),rolls=[])
        w.player.magic_immune=True
        self.assert_rejected_unchanged(w,engine,2,'magic_immune')

    def test_tempest_partial_allocation_and_distance_penalty(self):
        self.assertEqual(spells.tempest_allocation(10000,1.25,1,2),(7684,3842))
        self.assertEqual(spells.tempest_allocation(10000,.75,1,2),(7684,3842))
        self.assertEqual(spells.tempest_allocation(10000,.5,0,1),(0,0))
        self.assertEqual(spells.tempest_allocation(10000,1.5,0,0),(0,0))
        # With zero Dexterity/Strength, minRadius determines the full/partial ring.
        tempest=replace(baseline_spells()[2],min_radius=1.25)
        cfg=ScenarioConfig(player=replace(PlayerConfig(),dexterity=0,strength=0),
                           spells=(*baseline_spells()[:2],tempest))
        env,w,engine=self.scene(((51,50),(51,51),(52,51),(53,50)),config=cfg,rolls=[False]*4)
        engine.cast(3)
        self.assertEqual([e.target_id for e in engine.damage_events],[2,3,4])
        self.assertLess(engine.damage_events[2].raw_damage,engine.damage_events[0].raw_damage)
        self.assertEqual(w.monsters[5].hp,w.monsters[5].max_hp)
        spell=baseline_spells()[2]
        near=spells.calculate(spell,PlayerConfig(),14358,9463,((2,0),),caster_distance=1)
        far=spells.calculate(spell,PlayerConfig(),14358,9463,((2,0),),caster_distance=10)
        self.assertLess(far.total_damage,near.total_damage)
        self.assertEqual(near.radius,1.5)

    def test_dot_snapshot_ticks_expiry_and_no_extra_rng(self):
        env,w,engine=self.scene(rolls=[False,False])
        engine.cast(2)
        after_initial=w.monsters[2].hp
        self.assertEqual(w.effects[0].effect_id,'acid:1')
        self.assertEqual(w.effects[0].expires_at_ms,6000)
        engine.scheduler.advance(999,engine.dispatch)
        self.assertEqual(w.monsters[2].hp,after_initial)
        # Moving away and becoming immune does not change the source callback snapshot.
        w.monsters[2].x=90
        w.monsters[2].magic_immune=True
        w.player.mp=0
        engine.scheduler.advance(6000,engine.dispatch)
        self.assertEqual(w.monsters[2].hp,after_initial-5*939)
        self.assertEqual([e.time_ms for e in engine.damage_events if e.damage_type=='dot'],[1000,2000,3000,4000,5000])
        self.assertEqual(w.effects,[])
        self.assertEqual(len(engine.rng.calls),2)

    def test_rectangular_casting_limits_and_fallback(self):
        for dx,dy,valid in ((16,10,True),(-16,-10,True),(16,-10,True),(-16,10,True),
                            (17,0,False),(-17,0,False),(0,11,False),(0,-11,False)):
            with self.subTest(dx=dx,dy=dy):
                env,w,engine=self.scene(((50+dx,50+dy),),rolls=[False,False] if valid else [])
                self.assertEqual(targeting.in_casting_range(w.player,w.monsters[2]),valid)
                if valid:
                    self.assertTrue(engine.cast(1)[0])
                else:
                    self.assert_rejected_unchanged(w,engine,1,'no_target')
                    self.assertEqual(cooldowns.rejection(w.player,env.config.spells[0],0,w.monsters[2]),'out_of_range')
                    for spell in env.config.spells[1:]:
                        self.assertIs(targeting.select_target(w,spell),w.player)
        # A nearer but ineligible target must not hide an eligible corner target.
        env,w,engine=self.scene(((50,61),(66,60)),rolls=[False,False])
        self.assertTrue(engine.cast(1)[0])
        self.assertEqual(engine.cast_events[0].target_id,3)

    def test_acid_refresh_replaces_snapshot_and_old_deadlines(self):
        env,w,engine=self.scene(rolls=[False]*4)
        engine.cast(2)
        old_generation=w.effects[0].generation
        engine.scheduler.advance(5000,engine.dispatch)
        self.assertTrue(engine.cast(2)[0])
        self.assertEqual(len(w.effects),1)
        self.assertEqual(w.effects[0].effect_id,'acid:1')
        self.assertGreater(w.effects[0].generation,old_generation)
        self.assertEqual(w.effects[0].next_tick_ms,6000)
        self.assertEqual(w.effects[0].expires_at_ms,11000)
        self.assertLess(w.effects[0].tick_damage,939)  # New cast used remaining MP.
        tick_damage=w.effects[0].tick_damage
        hp=w.monsters[2].hp
        engine.scheduler.advance(6000,engine.dispatch)
        self.assertEqual(len(w.effects),1)  # Old expiry cannot delete the refresh.
        self.assertEqual(w.monsters[2].hp,hp-tick_damage)
        engine.scheduler.advance(11000,engine.dispatch)
        self.assertEqual(w.effects,[])
        self.assertEqual(w.monsters[2].hp,hp-5*tick_damage)
        self.assertEqual(sum(e['kind']=='effect_refreshed' for e in engine.trace),1)

    def test_acid_different_casters_stack_and_refresh_independently(self):
        env,w,engine=self.scene(rolls=[])
        second=replace(w.player,entity_id=99)
        original_entity=engine.entity
        with patch.object(engine,'entity',side_effect=lambda i: second if i==99 else original_entity(i)):
            engine.add_acid(env.config.spells[1],spells.Allocation(2,0,tick_damage=10),False)
            engine.add_acid(env.config.spells[1],spells.Allocation(2,0,tick_damage=20),True,source_id=99)
            engine.scheduler.advance(500,engine.dispatch)
            engine.add_acid(env.config.spells[1],spells.Allocation(2,0,tick_damage=30),False)
            self.assertEqual({e.effect_id for e in w.effects},{'acid:1','acid:99'})
            hp=w.monsters[2].hp
            engine.scheduler.advance(1000,engine.dispatch)
            self.assertEqual(w.monsters[2].hp,hp-20)  # Cancelled old caster-1 tick.
            engine.scheduler.advance(1500,engine.dispatch)
            self.assertEqual(w.monsters[2].hp,hp-50)
            self.assertEqual([e.attacker_id for e in engine.damage_events],[99,1])
            engine.scheduler.advance(6000,engine.dispatch)
            self.assertEqual([e.effect_id for e in w.effects],['acid:1'])
            engine.scheduler.advance(6500,engine.dispatch)
            self.assertEqual(w.effects,[])
        self.assertEqual(engine.rng.calls,[])

    def test_initial_and_tick_kills_cleanup(self):
        for slot in (1,2,3):
            env,w,engine=self.scene(rolls=[False,False])
            w.monsters[2].hp=1
            engine.cast(slot)
            self.assertEqual(w.kills,1)
            self.assertEqual(w.effects,[])
            self.assertEqual(w.monsters[2].respawn_at_ms,50000)
            self.assertNotIn((51,50),w.occupancy)
        env,w,engine=self.scene(rolls=[False,False])
        engine.cast(2)
        w.monsters[2].hp=1
        engine.scheduler.advance(6000,engine.dispatch)
        self.assertEqual(w.kills,1)
        self.assertEqual(w.monsters[2].respawn_at_ms,51000)
        self.assertEqual(w.effects,[])
        self.assertEqual(len(engine.death_events),1)
        self.assertEqual(len(engine.damage_events),2)

    def test_baseline_modifiers_explicitly_disabled(self):
        for name in ('arcane_bomb','sunburnt','tempest_meteor'):
            with self.assertRaises(ValueError): SpellRules(**{name:True})

    def test_seeded_replay_includes_casts_cooldowns_effects(self):
        cfg=ScenarioConfig(player=replace(PlayerConfig(),max_hp=10000000),
                           reward=RewardConfig(win_kills=100),timing=TimingConfig(regen_enabled=False))
        a,b=MysticSimEnv(config=cfg,trace=True),MysticSimEnv(config=cfg,trace=True)
        self.addCleanup(a.close);self.addCleanup(b.close)
        a.reset(seed=42);b.reset(seed=42)
        cast_count=tick_count=0
        with patch('builtins.open',side_effect=AssertionError('runtime file access')):
            for i in range(100):
                action=[6,5,7,4][i%4]
                ra,rb=a.step(action),b.step(action)
                assert_array_equal(ra[0],rb[0])
                self.assertEqual(ra[1:],rb[1:])
                self.assertEqual(a.world,b.world)
                cast_count+=len(ra[4]['cast_events'])
                tick_count+=sum(e['kind']=='effect_tick' for e in ra[4]['trace'])
        self.assertGreater(cast_count,0)
        self.assertGreater(tick_count,0)


if __name__ == '__main__':
    unittest.main()
