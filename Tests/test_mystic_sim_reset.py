import copy
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from Custom_enviornments.Mystic_Sim.config import (
    InnieConfig, PlayerConfig, RewardConfig, ScenarioConfig, TimingConfig, baseline_spells,
)
from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
from Custom_enviornments.Mystic_Sim.map_loader import DEFAULT_MAP_PATH, load_map53, normalize_map53
from Custom_enviornments.Mystic_Sim.observation import (
    direction_from_delta, direction_from_server, encode_observation,
    euclidean_distance, nearest_monsters, observation_distance,
)
from Custom_enviornments.Mystic_Sim.scenarios import build_scenario, sample_unoccupied
from Custom_enviornments.Mystic_Sim.state import Direction, MonsterState, PlayerState, WorldState
from Custom_enviornments.Test_Env.Env_conditions import encode_observation as live_encode

FIXTURES = Path(__file__).parent / "fixtures/simulation"


class MysticResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = normalize_map53(load_map53(DEFAULT_MAP_PATH))

    def test_normalized_map_preserves_geometry_and_metadata(self):
        self.assertEqual(len(self.map.spawn_boxes), 40)
        self.assertEqual(sum(b.quantity for b in self.map.spawn_boxes), 80)
        self.assertEqual((self.map.width, self.map.height, self.map.tile_width, self.map.tile_height),
                         (100, 100, 32, 32))
        self.assertEqual(self.map.parsed_balance_cap, 68)
        self.assertEqual(len(self.map.blocked_cells), 5088)
        self.assertEqual(self.map.excluded_npcs[0].template_id, 5399)
        box = next(b for b in self.map.spawn_boxes if b.object_id == 177)
        self.assertEqual((box.x, box.y, box.width, box.height), (40, 71, 10, 10))
        self.assertTrue(box.contains(49, 80))
        self.assertFalse(box.contains(50, 81))
        with self.assertRaises(FrozenInstanceError):
            box.x = 1
        with self.assertRaises(FrozenInstanceError):
            self.map.width = 50
        raw = load_map53(DEFAULT_MAP_PATH)
        raw["layers"][4]["objects"].reverse()
        self.assertEqual(normalize_map53(raw), self.map)
        self.assertEqual(DEFAULT_MAP_PATH.read_bytes(), (FIXTURES / "map53.json").read_bytes())

    def test_reset_invariants_across_seeds(self):
        env = MysticSimEnv()
        signatures = set()
        for seed in range(30):
            obs, info = env.reset(seed=seed)
            w = env.world
            signatures.add(tuple(info["npc_spawns"].items()))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(len(w.monsters), 80)
            self.assertEqual(len(w.occupancy), 81)
            self.assertEqual(set(w.occupancy.values()), {1, *w.monsters.keys()})
            self.assertTrue(45 <= w.player.x <= 55 and 40 <= w.player.y <= 50)
            self.assertEqual((w.player.hp, w.player.mp, w.player.class_spec), (9463, 15108, "Mystic"))
            self.assertEqual((w.time_ms, w.step_count, w.kills), (0, 0, 0))
            self.assertFalse(w.effects)
            self.assertFalse(w.player.cooldowns.slots)
            self.assertFalse(w.player.cooldowns.families)
            self.assertEqual((info["parsed_balance_cap"], info["npc_effective_level_override"]), (68, 150))
            for m in w.monsters.values():
                self.assertTrue(m.spawn_box.contains(m.x, m.y))
                self.assertTrue(0 <= m.x < 100 and 0 <= m.y < 100)
                self.assertEqual(w.occupancy[(m.x, m.y)], m.entity_id)
                self.assertEqual((m.hp, m.mp, m.stats.effective_level), (274599, 0, 150))
                self.assertEqual(m.stats.respawn_ms, 50000)
                self.assertIsNone(m.aggro_target)
                self.assertIsNone(m.respawn_at_ms)
                self.assertTrue(900 <= m.move_interval_ms <= 1100)
                self.assertEqual(m.move_interval_ms, m.next_move_ms)
            self.assertEqual(len(w.events), 81)
            self.assertEqual(w.events, sorted(w.events))
            self.assertEqual({e.sequence for e in w.events}, set(range(81)))
            regen = [e for e in w.events if e.kind == "regeneration"]
            self.assertEqual([(e.entity_id, e.due_ms) for e in regen], [(1, 2000)])
            for event in w.events:
                if event.kind == "npc_update":
                    m = w.monsters[event.entity_id]
                    self.assertEqual(event.due_ms, min(m.next_move_ms, m.next_attack_ms, m.next_aggro_ms))
                    self.assertEqual((m.next_attack_ms, m.next_aggro_ms), (1000, 1500))
        self.assertEqual(len(signatures), 30)

    def test_seed_replay_and_reset_cleanup(self):
        env, other = MysticSimEnv(), MysticSimEnv()
        first, info = env.reset(seed=42)
        original = copy.deepcopy(env.world)
        env.world.player.hp = 1
        env.world.player.cooldowns.slots[1] = 1000
        env.world.monsters[2].hp = 0
        env.world.effects.append("stale")
        env.world.events.clear()
        env.world.occupancy.clear()
        env.world.kills = 4
        second, again = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info, again)
        self.assertEqual(env.world, original)
        other.reset(seed=42)
        self.assertEqual(env.world, other.world)
        env.world.player.cooldowns.slots[1] = 10
        self.assertFalse(other.world.player.cooldowns.slots)
        info["npc_spawns"].clear()
        self.assertEqual(len(env.world.monsters), 80)
        env.reset()  # Continuing RNG, not re-seeding to 42.
        self.assertNotEqual(env.world, original)

    def test_reset_and_observation_do_not_use_io(self):
        env = MysticSimEnv()
        with patch("builtins.open", side_effect=AssertionError("file I/O")), \
                patch.object(Path, "open", side_effect=AssertionError("file I/O")), \
                patch("socket.socket", side_effect=AssertionError("socket")), \
                patch("time.sleep", side_effect=AssertionError("sleep")):
            obs, info = env.reset(seed=5)
            intervals = [m.move_interval_ms for m in env.world.monsters.values()]
            for _ in range(3):
                np.testing.assert_array_equal(obs, encode_observation(env.world))
            self.assertEqual(intervals, list(info["npc_move_intervals_ms"].values()))

    def test_placement_full_box_and_rejection_draw_order(self):
        class StubRng:
            def __init__(self, values):
                self.values = iter(values)
                self.calls = []
            def integers(self, low, high):
                self.calls.append((low, high))
                return next(self.values)
        rng = StubRng([10, 20, 11, 20])
        self.assertEqual(sample_unoccupied(rng, 10, 20, 2, 1, {(10, 20): 1}), (11, 20))
        self.assertEqual(rng.calls, [(10, 12), (20, 21)] * 2)
        full = StubRng([])
        with self.assertRaisesRegex(ValueError, "no free cell"):
            sample_unoccupied(full, 0, 0, 1, 1, {(0, 0): 1})
        self.assertEqual(full.calls, [])

    def test_spawn_rng_speed_endpoints_and_order(self):
        class StubRng:
            def __init__(self):
                self.values = iter([45, 40, 10, 31, 900, 11, 31, 1100])
                self.calls = []
            def integers(self, low, high):
                self.calls.append((low, high))
                return next(self.values)
        rng = StubRng()
        small_map = replace(self.map, spawn_boxes=self.map.spawn_boxes[:1])
        world = build_scenario(small_map, ScenarioConfig(), rng)
        self.assertEqual([m.move_interval_ms for m in world.monsters.values()], [900, 1100])
        self.assertEqual(rng.calls, [(45, 56), (40, 51), (10, 20), (31, 41), (900, 1101),
                                     (10, 20), (31, 41), (900, 1101)])

    def test_terrain_disabled_at_reset(self):
        # All cells marked blocked still leave placement legal in the open profile.
        map_def = replace(self.map, blocked_cells=frozenset((x, y) for x in range(100) for y in range(100)))
        world = build_scenario(map_def, ScenarioConfig(), np.random.default_rng(3))
        self.assertEqual(len(world.occupancy), 81)

    def test_observation_padding_ties_dead_monsters_and_directions(self):
        env = MysticSimEnv()
        env.reset(seed=1)
        w = env.world
        w.player.x = w.player.y = 50
        w.player.hp = 0
        w.player.mp = w.player.max_mp * 2
        a, b, c = (w.monsters[i] for i in (2, 3, 4))
        a.x, a.y = 51, 50
        b.x, b.y = 50, 49
        c.x, c.y, c.hp = 50, 50, 0
        w.monsters = {4: c, 3: b, 2: a}
        obs = encode_observation(w)
        self.assertEqual([m.entity_id for m in nearest_monsters(w)], [2, 3])
        np.testing.assert_array_equal(obs[:6], [50, 50, 0, 0, 1, 53])
        np.testing.assert_array_equal(obs[6:14], [1, 3, 1, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(obs[14:], np.zeros(12))
        w.monsters.clear()
        np.testing.assert_array_equal(encode_observation(w)[6:], np.zeros(20))
        for dx, dy, expected in [(0,0,4),(1,0,3),(-1,0,2),(0,-1,0),(0,1,1),(1,1,1),(-1,-1,0)]:
            self.assertEqual(int(direction_from_delta(dx, dy)), expected)
        self.assertEqual([int(direction_from_server(i)) for i in range(4)], [2, 3, 0, 1])
        self.assertEqual(observation_distance(0, 0, 1, 1), 2)
        self.assertAlmostEqual(euclidean_distance(0, 0, 1, 1), 2**.5)
        self.assertEqual(euclidean_distance(0, 0, 1, 0), 1)
        with self.assertRaises(ValueError):
            direction_from_server(4)

    def test_state_encoder_matches_full_state_fixture(self):
        payload = json.loads((FIXTURES / "example_full_state.json").read_text())
        p = payload["player"]
        player_config = replace(PlayerConfig(), max_hp=p["maxHp"], max_mp=p["maxMp"])
        player = PlayerState(p["id"], p["mapX"], p["mapY"], p["hp"], p["mp"],
                             player_config, facing=Direction.LEFT)
        world = WorldState(replace(self.map, map_id=3471), player)
        for e in payload["entities"]:
            if e.get("type") != "monster":
                continue
            stats = replace(InnieConfig(), max_hp=e["maxHp"], max_mp=e.get("maxMp", 0))
            world.monsters[e["id"]] = MonsterState(e["id"], e["mapX"], e["mapY"], e["hp"], e.get("mp",0),
                stats, self.map.spawn_boxes[0], 0, 1000, 1000, 1000, 1500)
        np.testing.assert_array_equal(encode_observation(world), live_encode(payload))
        expected = json.loads((FIXTURES / "observation_expected.json").read_text())
        self.assertEqual([m.entity_id for m in nearest_monsters(world)], expected["selected_ids"])

    def test_config_validation_and_immutability(self):
        cases = [lambda: PlayerConfig(max_hp=0), lambda: PlayerConfig(max_mp=-1),
                 lambda: InnieConfig(attack_ms=0), lambda: InnieConfig(move_ms=(1100,900)),
                 lambda: InnieConfig(respawn_ms=0), lambda: TimingConfig(step_ms=0),
                 lambda: RewardConfig(win_kills=0), lambda: ScenarioConfig(player_spawn_x=(99,100)),
                 lambda: ScenarioConfig(player_spawn_y=(50,40)), lambda: ScenarioConfig(width=99),
                 lambda: ScenarioConfig(spells=(baseline_spells()[0],)*3),
                 lambda: ScenarioConfig(terrain_collision=True), lambda: ScenarioConfig(player={}),
                 lambda: replace(baseline_spells()[0], cooldown_ms=-1),
                 lambda: replace(baseline_spells()[1], tick_interval_ms=0)]
        for create in cases:
            with self.assertRaises(ValueError):
                create()
        config = ScenarioConfig()
        with self.assertRaises(FrozenInstanceError):
            config.player.max_hp = 100
        env = MysticSimEnv(config=config)
        for options in ({"profile":"missing"}, {"unknown":1}, []):
            with self.assertRaises(ValueError):
                env.reset(seed=1, options=options)
        env.reset(seed=1, options={"profile":config.profile})

    def test_config_tracks_manifest_spell_properties(self):
        path = DEFAULT_MAP_PATH.parent.parent / "mechanics_manifest.yaml"
        manifest = yaml.safe_load(path.read_text())
        mapping = {"cooldown_ms":"cooldown", "mp_cost":"mpCost", "mana_consumption":"manaConsumption",
                   "mana_factor":"manaFactor", "damage_percent":"damagePercent", "target_mode":"targetMode",
                   "family":"family", "family_cooldown_ms":"familyCooldown", "duration_ms":"duration",
                   "tick_interval_ms":"interval", "initial_damage_percent":"initialDamagePercent",
                   "tick_percent":"tickPercent", "min_radius":"minRadius", "max_radius":"maxRadius",
                   "level":"level", "cast_level":"castLevel", "effect_id":"effectId", "effect_level":"effectLevel"}
        for spell in baseline_spells():
            for field, key in mapping.items():
                if key in manifest["spells"][spell.spell_id]:
                    self.assertEqual(getattr(spell,field), manifest["spells"][spell.spell_id][key]["value"])


if __name__ == "__main__":
    unittest.main()
