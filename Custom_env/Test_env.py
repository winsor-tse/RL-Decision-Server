import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
from Custom_env import Parse_data
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
        self.next_state = np.zeros(OBS_SIZE, dtype=np.float32)  # instead of 21
        self.current_step = 0
        self.max_steps = 20 #Only used for testing main below, the max steps is done in server
        #Intialize ZMQ and sockets
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(ZMQ_BIND_URL)

    def _get_obs(self):
        return self.next_state

    #This is currently fine but cleanRL's native info looks at final_observation and final_info
    #Not sure how to handle this customly
    def _get_info(self):
        return {"current_step": self.current_step, "next_state": self.next_state}
    
    def _send_action(self, action_idx):
        return random.choice(ACTIONS)

    def reset(self):
        message = self.socket.recv_json()
        print("RESET Received:", message)
        message_type = message.get("type")
        if message_type == "ai_tick":
            response = {
                "type": "ai_result",
                "requestId": message.get("requestId"),
                "move": "direction:up", #handle None in JS or do nothing
                "reset": True,
                "serverTime": time.time()
            }
        else:
            response = {
                "type": "error",
                "requestId": message.get("requestId"),
                "error": f"Unknown message type: {message_type}"
            }
        self.socket.send_json(response)
        world_state = message.get("worldState", {})
        self.next_state = np.zeros(OBS_SIZE, dtype=np.float32) #needs to be nothing
        real_next_state = Parse_data.parse_observation(world_state, OBS_SIZE)
        self.current_step = 0
        print(f"RESET")
        return real_next_state, self._get_info()

    def step(self, action):
        message = self.socket.recv_json()
        #print("STEP Received:", message)
        message_type = message.get("type")
        world_state = message.get("worldState", {})
        if message_type == "ai_tick":
            response = {
                "type": "ai_result",
                "requestId": message.get("requestId"),
                "move": self.Actions[int(action)],
                "reset": False,
                "serverTime": time.time()
            }
        else:
            response = {
                "type": "error",
                "requestId": message.get("requestId"),
                "error": f"Unknown message type: {message_type}"
            }
        self.socket.send_json(response)
        self.current_step += 1
        real_next_state = Parse_data.parse_observation(world_state, OBS_SIZE)
        #TODO: need to be really careful here actually b/c one edge case is if we terminate or reset epoch
        # we can get obs from previous epochs and continually terminating or give false rewards
        reward = Parse_data.get_reward(real_next_state, action, self.next_state) #self.current_state is previous
        terminated = Parse_data.get_termination(real_next_state, self.next_state)
        truncated = Parse_data.get_truncated(real_next_state, self.next_state, self.current_step)
        self.next_state = real_next_state
        return self.next_state, reward, terminated, False, self._get_info()

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