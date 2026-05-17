import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
import Parse_data
import zmq

ACTIONS = ["up", "down", "left", "right", "direction:up", "direction:down", "direction:left", "direction:right", "attack", "castSpell:1", "castSpell:2", "castSpell:3"]
ZMQ_BIND_URL = "tcp://127.0.0.1:5555"
OBS_PLAYER_SIZE = 5
OBS_ENEMY_SIZE = 4
MAX_ENEMIES = 2
OBS_SIZE = OBS_PLAYER_SIZE + OBS_ENEMY_SIZE * MAX_ENEMIES  # 13

#TODO: test this enviornment

class TestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.Actions = ACTIONS
        self.single_action_space = spaces.Discrete(len(self.Actions))
        self.single_observation_space = spaces.Box(low=0, high=math.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.current_state = np.zeros(OBS_SIZE, dtype=np.float32)  # instead of 21
        self.current_step = 0
        self.max_steps = 20
        #Intialize
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(ZMQ_BIND_URL)

    def _get_obs(self):
        return self.current_state

    def _get_info(self):
        return {"current_step": self.current_step, "current_state": self.current_state}
    
    def _send_action(self, action_idx):
        return random.choice(ACTIONS)

    def reset(self):
        message = env.socket.recv_json()
        print("RESET Received:", message)
        message_type = message.get("type")
        if message_type == "ai_tick":
            response = {
                "type": "ai_result",
                "requestId": message.get("requestId"),
                "move": "direction:up", #handle None in JS
                "reset": True,
                "serverTime": time.time()
            }
        else:
            response = {
                "type": "error",
                "requestId": message.get("requestId"),
                "error": f"Unknown message type: {message_type}"
            }
        env.socket.send_json(response)
        world_state = message.get("worldState", {})
        self.current_state = Parse_data.parse_observation(world_state, OBS_SIZE)
        return self.current_state, self._get_info()

    def step(self, action):
        message = env.socket.recv_json()
        print("STEP Received:", message)
        message_type = message.get("type")
        world_state = message.get("worldState", {})
        if message_type == "ai_tick":
            response = {
                "type": "ai_result",
                "requestId": message.get("requestId"),
                "move": self.Actions[action],
                "reset": False,
                "serverTime": time.time()
            }
        else:
            response = {
                "type": "error",
                "requestId": message.get("requestId"),
                "error": f"Unknown message type: {message_type}"
            }
        env.socket.send_json(response)
        self.current_step += 1
        self.current_state = Parse_data.parse_observation(world_state, OBS_SIZE)
        return self.current_state, 0, False, False, self._get_info()

# Example usage:
if __name__ == "__main__":
    env = TestEnv()
    #bs, info = env.reset()
    #print("Initial Observation:", obs)
    for _ in range(env.max_steps):
        #DO AI stuff here
        action = np.random.randint(len(ACTIONS)) #replaced with DRL categorical action eventually
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    env.close()