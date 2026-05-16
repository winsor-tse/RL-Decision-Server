import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
import Parse_data

ACTIONS = ["up", "down", "left", "right", "direction:up", "direction:down", "direction:left", "direction:right", "attack", "castSpell:1", "castSpell:2", "castSpell:3"]
ZMQ_BIND_URL = "tcp://127.0.0.1:5555"

#TODO: the current world state is
"""
Received: {'type': 'ai_tick', 'requestId': '5a0a6841-082c-4c5a-b270-5d4c27cb2bbd', 'worldState': {'timestamp': 1778889927696, '
player': {'id': 7, 'name': 'testsd', 'mapX': 18, 'mapY': 29, 'direction': 'up', 'hp': 1002383, 'maxHp': 1002383, 'mp': 1003118, 'maxMp': 1003118}, 
'entities': [{'id': 2, 'name': 'Sage of Welcoming', 'type': 'monster', 'isCurrentPlayer': False, 'mapX': 14, 'mapY': 13, 'hp': 500000, 'maxHp': 500000, 'mp': 0, 'maxMp': 0, 'distance': 20}, 
{'id': 7, 'name': 'testsd', 'type': 'player', 'isCurrentPlayer': True, 'mapX': 18, 'mapY': 29, 'hp': 1002383, 'maxHp': 1002383, 'mp': 1003118, 'maxMp': 1003118, 'distance': 0}]},
 'pageUrl': 'http://127.0.0.1:8080/?server=test.yugensaga.com', 'timestamp': 1778889927696}
"""
#parse into and save as spaces.Box mapX, mapY, direction(0-4), percentageHP (hp/maxHP), percentageMP (mp/maxMP), 

class TestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.Actions = ACTIONS
        self.single_action_space = spaces.Discrete(len(self.Actions))
        self.single_observation_space = spaces.Box(low=0, high=math.inf, shape=(21,), dtype=np.float32)
        self.current_state = np.zeros(21, dtype=np.float32)  # instead of 21
        self.current_step = 0
        self.max_steps = 20
        #Intialize
        self.context = zmq.Context.instance()
        self.socket = context.socket(zmq.REP)
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
        self.current_state = Parse_data.parse_observation(world_state)
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
        self.current_state = Parse_data.parse_observation(world_state)
        return self.current_state, 0, false, false, info

# Example usage:
if __name__ == "__main__":
    env = TestEnv()
    obs, info = env.reset()
    print("Initial Observation:", obs)
    for _ in range(env.max_steps):
        #DO AI stuff here
        action = np.random.randint(len(ACTIONS)) #replaced with DRL categorical action eventually
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    env.close()