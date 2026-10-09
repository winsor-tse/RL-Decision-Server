from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from Custom_enviornments.Mystic_Sim import combat
from Custom_enviornments.Mystic_Sim.config import ScenarioConfig, PlayerConfig, InnieConfig, GearConfig, TimingConfig
from Custom_enviornments.Mystic_Sim.engine import Engine
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.scaled_calcs import NPCRecursiveDamage, expected_regression_dps, scale_innie
from Custom_enviornments.Mystic_Sim.state import Direction, TimedEffect


class ScriptedRng:
    def __init__(self, values): self.values=iter(values); self.calls=[]; self.entity_id=None
    def chance(self, probability, purpose):
        self.calls.append((purpose,probability))
        return next(self.values)


class CombatTests(unittest.TestCase):
    def scene(self, config=None):
        env=MysticSimEnv(config=config,trace=True)
        env.reset(seed=14)
        w=env.world
        npc=w.monsters[2]
        w.monsters={2:npc}
        w.events.clear()
        p=w.player
        p.x,p.y,p.facing=50,50,Direction.RIGHT
        npc.x,npc.y,npc.facing=51,50,Direction.LEFT
        npc.next_move_ms=1000000
        npc.spawn_box=replace(npc.spawn_box,x=50,y=50)
        w.occupancy={(50,50):1,(51,50):2}
        env.engine=Engine(w,env.config,env.np_random,trace=True)
        return env,w,p,npc

    def test_numeric_regression_values(self):
        values=json.loads((Path(__file__).parent/'fixtures/simulation/combat_expected.json').read_text())
        lookup=NPCRecursiveDamage()
        for row in values['scaling']:
            stats=scale_innie(InnieConfig(effective_level=row['level']))
            self.assertEqual((stats.max_hp,stats.raw_damage),(row['hp'],row['raw_damage']))
            self.assertAlmostEqual(lookup.get_damage(row['level']),row['recursive_damage'],places=10)
            self.assertAlmostEqual(expected_regression_dps(row['level']),row['dps'])
        old=list(lookup.lookup)
        lookup.get_damage(135)
        self.assertEqual(old,lookup.lookup)
        self.assertEqual(lookup.get_damage(-1),0)
        for row in values['factors']:
            self.assertAlmostEqual(combat.stat_factor(row['value']),row['factor'],places=12)
        p=PlayerConfig(); golden=values['player']
        self.assertAlmostEqual(combat.melee_crit_chance(p.dexterity),golden['melee_chance'],places=12)
        self.assertAlmostEqual(combat.spell_crit_chance(p.dexterity,p.wisdom),golden['spell_chance'],places=12)
        self.assertAlmostEqual(combat.crit_multiplier(p),golden['melee_multiplier'],places=12)
        self.assertAlmostEqual(combat.crit_multiplier(p,True),golden['magic_multiplier'],places=12)
        env,w,player,npc=self.scene()
        baseline_stats=asdict(npc.stats)
        for row in values['mitigation']:
            stats=dict(baseline_stats); stats['ac']=row['ac']
            npc.stats=SimpleNamespace(**stats)
            damage,blocked=combat.mitigate(row['damage'],player,npc,ScriptedRng([row['blocked']]),magic=True)
            self.assertEqual(damage,row['expected'])
        for row in values['player_mitigation']:
            self.assertEqual(combat.mitigate(3549,npc,player,ScriptedRng([row['blocked']]),magic=True)[0],row['expected'])
        for row in values['aoe']:
            self.assertEqual(combat.apply_aoe(10000,row['count']),row['expected'])
        with self.assertRaises(ValueError): combat.apply_aoe(100,0)

    def test_npc_attack_dodge_block_normal_rng_order(self):
        for rolls,expected,purposes in [([True],0,['dodge']),([False,False],1828,['dodge','block']),
                                       ([False,True],1097,['dodge','block'])]:
            env,w,p,npc=self.scene()
            rng=ScriptedRng(rolls); env.engine.rng=rng
            event=env.engine.melee_attack(npc,p)
            self.assertEqual(event.damage,expected)
            self.assertEqual(p.hp,p.max_hp-expected)
            self.assertEqual([x[0] for x in rng.calls],purposes)
            self.assertFalse(event.crit)

    def test_facing_and_magic_immunity(self):
        env,w,p,npc=self.scene()
        self.assertEqual(combat.facing_bonus(p,npc),1)
        npc.facing=Direction.RIGHT
        self.assertEqual(combat.facing_bonus(p,npc),1.5)
        npc.facing=Direction.UP
        self.assertEqual(combat.facing_bonus(p,npc),1.25)
        self.assertTrue(combat.can_hit(p,npc,magic=True))
        npc.magic_immune=True
        self.assertFalse(combat.can_hit(p,npc,magic=True))
        self.assertFalse(combat.can_hit(p,p))
        self.assertFalse(combat.can_hit(npc,npc))
        env.engine.rng=ScriptedRng([])
        self.assertIsNone(env.engine.magic_damage(p,npc,10000))

    def test_raw_spell_and_mitigation_use_independent_rolls(self):
        env,w,p,npc=self.scene()
        rng=ScriptedRng([True,False]); env.engine.rng=rng
        raw,crit=combat.raw_spell_damage(1000,p.stats,rng)
        self.assertEqual(raw,int(1000*combat.stat_factor(578)*combat.crit_multiplier(p.stats,True)))
        event=env.engine.magic_damage(p,npc,raw,crit=crit)
        self.assertTrue(event.crit)
        self.assertEqual([c[0] for c in rng.calls],['spell_crit','block'])
        self.assertEqual(npc.aggro_target,p.entity_id)

    def test_death_once_cleans_state_and_respawns_same_id(self):
        env,w,p,npc=self.scene()
        engine=env.engine
        engine.queue_npc(npc)
        engine.schedule_effect(TimedEffect('acid',1,2,1000,6000,1000))
        npc.aggro_target=1
        event=engine.apply_damage(p,npc,npc.hp+100)
        self.assertEqual(event.damage,npc.max_hp)
        self.assertEqual(w.kills,1)
        self.assertNotIn((51,50),w.occupancy)
        self.assertFalse(w.effects)
        self.assertIsNone(npc.aggro_target)
        self.assertEqual(npc.respawn_at_ms,50000)
        self.assertIsNone(engine.apply_damage(p,npc,100))
        self.assertEqual(len(engine.death_events),1)
        engine.scheduler.advance(49999,engine.dispatch)
        self.assertFalse(npc.alive)
        engine.scheduler.advance(50000,engine.dispatch)
        self.assertTrue(npc.alive)
        self.assertIs(w.monsters[2],npc)
        self.assertEqual((npc.hp,npc.mp),(npc.max_hp,npc.max_mp))
        self.assertEqual(w.occupancy[(npc.x,npc.y)],2)
        self.assertTrue(npc.spawn_box.contains(npc.x,npc.y))
        self.assertEqual(npc.next_move_ms,50000+npc.move_interval_ms)
        self.assertEqual((npc.next_attack_ms,npc.next_aggro_ms),(51000,51500))
        self.assertEqual(len(engine.respawn_events),1)
        self.assertEqual(w.kills,1)
        draws=[r['purpose'] for r in engine.trace if r['kind']=='rng']
        self.assertEqual(draws[-1],'respawn_move_interval')
        self.assertEqual(draws.count('respawn_move_interval'),1)
        self.assertTrue(900<=npc.move_interval_ms<=1100)

    def test_full_spawn_box_retries_next_boundary_without_draws(self):
        env,w,p,npc=self.scene()
        engine=env.engine
        w.time_ms=37
        engine.apply_damage(p,npc,npc.hp)
        box=npc.spawn_box
        for x in range(box.x,box.x+box.width):
            for y in range(box.y,box.y+box.height): w.occupancy[(x,y)]=999
        engine.scheduler.advance(50037,engine.dispatch)
        self.assertEqual(npc.respawn_at_ms,50200)
        self.assertFalse(any(r['kind']=='rng' for r in engine.trace))
        del w.occupancy[(59,59)]
        engine.scheduler.advance(50200,engine.dispatch)
        self.assertEqual((npc.x,npc.y),(59,59))
        self.assertTrue(npc.alive)

    def test_player_death_terminates_and_prevents_regeneration(self):
        env,w,p,npc=self.scene()
        npc.aggro_target=1
        env.engine.apply_damage(npc,p,p.hp)
        self.assertNotIn((50,50),w.occupancy)
        self.assertIsNone(npc.aggro_target)
        env.engine.regenerate()
        self.assertEqual(p.hp,0)
        result=env.step(4)
        self.assertTrue(result[2])
        self.assertEqual(result[4]['episode_outcome'],'loss')

    def test_regeneration_rate_interval_toggle_clamp_and_formula(self):
        env,w,p,npc=self.scene()
        p.hp-=1000; p.mp=0
        env.engine.scheduler.schedule(2000,'regeneration',1)
        env.engine.scheduler.advance(1999,env.engine.dispatch)
        self.assertEqual((p.hp,p.mp),(8463,0))
        env.engine.scheduler.advance(2000,env.engine.dispatch)
        self.assertEqual((p.hp,p.mp),(8689,1664))
        p.hp=p.max_hp-1; p.mp=p.max_mp-1
        env.engine.scheduler.advance(4000,env.engine.dispatch)
        self.assertEqual((p.hp,p.mp),(p.max_hp,p.max_mp))
        env,w,p,npc=self.scene(ScenarioConfig(timing=TimingConfig(regen_enabled=False)))
        p.hp=100;p.mp=10
        env.engine.regenerate()
        self.assertEqual((p.hp,p.mp),(100,10))
        config=ScenarioConfig(player=PlayerConfig(base_hp=9000,base_mp=10000,
            hp_regen_override=None,mp_regen_override=None),timing=TimingConfig(regen_ms=1000))
        env,w,p,npc=self.scene(config)
        p.hp=p.mp=0;p.hp=1
        env.engine.regenerate()
        self.assertEqual(p.hp,1+combat.regen_amount(100,9463,9000))
        self.assertEqual(p.mp,combat.regen_amount(578,15108,10000))
        self.assertEqual(combat.regen_amount(0,100,0),2) # C# midpoint-to-even
        self.assertEqual(combat.regen_amount(0,140,0),4)

    def test_gear_disabled_and_enabled_attack_cadence(self):
        env,w,p,npc=self.scene()
        rng=ScriptedRng([]);env.engine.rng=rng
        self.assertEqual(env.step(4)[4]['action_failure_reason'],'gear_disabled')
        self.assertEqual(rng.calls,[])
        cfg=ScenarioConfig(gear_enabled=True,gear=GearConfig(100,1000))
        env,w,p,npc=self.scene(cfg)
        rng=ScriptedRng([False,True,False]);env.engine.rng=rng
        result=env.step(4)
        self.assertTrue(result[4]['action_applied'])
        self.assertTrue(result[4]['damage_events'][0]['crit'])
        self.assertEqual([r[0] for r in rng.calls],['dodge','melee_crit','block'])
        self.assertEqual(p.next_attack_ms,1000)
        self.assertEqual(env.step(4)[4]['action_failure_reason'],'cooldown')
        with self.assertRaises(ValueError): ScenarioConfig(gear_enabled=True)
        with self.assertRaises(ValueError): ScenarioConfig(gear=object())
        with self.assertRaises(ValueError): ScenarioConfig(gear=GearConfig(100,1000))

    def test_same_time_lethal_cancels_attack_and_magic_dead_no_rng(self):
        env,w,p,npc=self.scene();engine=env.engine
        npc.aggro_target=1;npc.next_attack_ms=1000
        engine.scheduler.schedule(1000,'kill',2)
        engine.queue_npc(npc,1000)
        def dispatch(event):
            if event.kind=='kill': engine.apply_damage(p,npc,npc.hp)
            else: engine.dispatch(event)
        engine.scheduler.advance(1000,dispatch)
        self.assertEqual(p.hp,p.max_hp)
        engine.rng=ScriptedRng([])
        self.assertIsNone(engine.magic_damage(p,npc,10000))
        self.assertEqual(engine.rng.calls,[])


if __name__=='__main__': unittest.main()
