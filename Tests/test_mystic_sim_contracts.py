import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import yaml

from Custom_enviornments.Mystic_Sim import ENV_ID
from Custom_enviornments.Mystic_Sim.actions import (
    ACTIONS, Action, action_from_label, action_label, contract_metadata,
    remap_legacy_bc, validate_contract,
)
from Custom_enviornments.Mystic_Sim.artifacts import checkpoint_envelope, checkpoint_state
from Custom_enviornments.Mystic_Sim.map_loader import load_map53, validate_map53
from Custom_enviornments.Test_Env.Env_conditions import encode_observation, observation_monsters
from Offline.remap_mystic_bc import migrate

FIXTURES = Path(__file__).parent / "fixtures" / "simulation"


class MysticPhaseZeroTests(unittest.TestCase):
    def test_action_roundtrip_and_rejected_values(self):
        self.assertEqual(len(Action), 8)
        for action in Action:
            self.assertEqual(action_from_label(action_label(action)), action)
        self.assertEqual(ACTIONS[:4], ("up", "down", "left", "right"))
        for invalid in [-1, 8, True, 1.5, "1"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                action_label(invalid)
        with self.assertRaises(ValueError):
            action_from_label("castSpell:5")

    def test_legacy_bc_remap(self):
        self.assertEqual(remap_legacy_bc(range(8)), [0, 2, 3, 1, 4, 5, 6, 7])
        for invalid in [8, 9, 10, -1, True, 1.5]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                remap_legacy_bc([invalid])

    def test_dataset_conversion_preserves_alignment_and_source(self):
        source = FIXTURES / "legacy_bc.json"
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new.json"
            migrate(source, destination)
            result = json.loads(destination.read_text())
            validate_contract(result)
            self.assertEqual(result["dataset_id"], "mystic/BC-v1")
            self.assertEqual(result["episodes"][0]["actions"], [0, 2, 3, 1, 4, 5, 6, 7])
            self.assertEqual(result["episodes"][0]["observations"], list(range(9)))
            with self.assertRaises(ValueError):
                migrate(source, destination)
            bad = json.loads(before)
            bad["episodes"][0]["actions"][0] = 8
            bad_path = Path(directory) / "bad.json"
            bad_path.write_text(json.dumps(bad))
            rejected = Path(directory) / "rejected.json"
            with self.assertRaisesRegex(ValueError, "removed action"):
                migrate(bad_path, rejected)
            self.assertFalse(rejected.exists())
        self.assertEqual(source.read_bytes(), before)

    def test_checkpoint_metadata_rejects_legacy_and_wrong_order(self):
        state = {"weights": [1, 2, 3]}
        self.assertEqual(checkpoint_state(checkpoint_envelope(state)), state)
        with self.assertRaisesRegex(ValueError, "legacy"):
            checkpoint_state(state)
        envelope = checkpoint_envelope(state)
        envelope["contract"]["actions"][1:4] = ["left", "right", "down"]
        with self.assertRaisesRegex(ValueError, "actions"):
            checkpoint_state(envelope)
        for key in contract_metadata():
            metadata = contract_metadata()
            del metadata[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(metadata)

    def test_full_state_golden_is_pure(self):
        world = json.loads((FIXTURES / "example_full_state.json").read_text())
        expected = json.loads((FIXTURES / "observation_expected.json").read_text())
        before = copy.deepcopy(world)
        with patch("builtins.open", side_effect=AssertionError("Unexpected file I/O")), \
                patch.object(Path, "open", side_effect=AssertionError("Unexpected file I/O")):
            obs = encode_observation(world)
            wrapped = encode_observation({"worldState": world})
        np.testing.assert_array_equal(obs, np.array(expected["observation"], dtype=np.float32))
        np.testing.assert_array_equal(obs, wrapped)
        self.assertEqual(obs.shape, tuple(expected["shape"]))
        self.assertEqual(str(obs.dtype), expected["dtype"])
        self.assertEqual([m["id"] for m in observation_monsters(world)[:5]], expected["selected_ids"])
        self.assertEqual(world, before)

    def test_fixture_hashes(self):
        for name, info in json.loads((FIXTURES / "provenance.json").read_text()).items():
            self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), info["sha256"])

    def test_real_map_and_corruptions(self):
        data = load_map53(FIXTURES / "map53.json")
        self.assertEqual(validate_map53(data)["innie_count"], 80)
        cases = [
            ("width", lambda d: d.update(width=99)),
            ("tilewidth", lambda d: d.update(tilewidth=16)),
            ("layers", lambda d: d.update(layers=[])),
            ("properties.balanceCap", lambda d: d["properties"][0].update(value=150)),
            ("quantity", lambda d: d["layers"][4]["objects"][1]["properties"][2].update(value=3)),
            ("Innie boxes", lambda d: d["layers"][4]["objects"].pop(1)),
            ("multiple of 32", lambda d: d["layers"][4]["objects"][1].update(x=321)),
            ("10000", lambda d: d["layers"][0]["data"].pop()),
        ]
        for message, corrupt in cases:
            invalid = copy.deepcopy(data)
            corrupt(invalid)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_map53(invalid)
        for invalid in [None, [], {"layers": None}]:
            with self.assertRaises(ValueError):
                validate_map53(invalid)

    def test_registration_has_no_live_socket(self):
        env = gym.make(ENV_ID)
        self.assertEqual(env.action_space.n, 8)
        self.assertEqual(env.observation_space.shape, (26,))
        self.assertNotIn("BaseEnv", [base.__name__ for base in type(env.unwrapped).__mro__])
        with self.assertRaisesRegex(NotImplementedError, "Phase 1"):
            env.reset(seed=1)
        env.close()

    def test_manifest_corrections_and_provenance(self):
        path = Path(__file__).parents[1] / "Custom_enviornments/Mystic_Sim/mechanics_manifest.yaml"
        manifest = yaml.safe_load(path.read_text())
        parameters = manifest["parameters"]
        for key, expected in {"respawn_ms": 50000, "innie_move_draws_per_life": 1,
                              "npc_effective_level": 150, "parsed_balance_cap": 68,
                              "fixed_mp_cost_before_spell": True}.items():
            self.assertEqual(parameters[key]["value"], expected)
        for record in list(parameters.values()) + [r for s in manifest["spells"].values() for r in s.values()]:
            self.assertTrue(record["source"])
            self.assertIn(record["provenance"], {"source", "developer", "fixture", "derived", "simulator_rule"})
        self.assertEqual(manifest["spells"][418]["manaConsumption"]["value"], .2)


if __name__ == "__main__":
    unittest.main()
