"""Headless Mystic simulator with seeded reset and deterministic movement steps."""
from gymnasium.envs.registration import register, registry

ENV_ID = "YugenSaga/MysticSim-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="Custom_enviornments.Mystic_Sim.env:MysticSimEnv",
             max_episode_steps=256)
