import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Parse_data
import time
import requests
import math
from fastapi import FastAPI, Request
import uvicorn
from dummy_api import run as start_dummy_server
import multiprocessing

app = FastAPI()

last_action = {}
last_observation = {}

DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "idle": 4  # if needed
}

ACTIONS = ['up', 'down', 'left', 'right', 'attack']

class CustomBlankEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.Actions = ACTIONS
        self.single_action_space = spaces.Discrete(len(self.Actions))
        self.single_observation_space = spaces.Box(low=0, high=math.inf, shape=(21,), dtype=np.float32)
        self.current_state = np.zeros(21, dtype=np.float32)  # instead of 21
        self.current_step = 0
        self.max_steps = 100
        self.timeout = 1000
        self.api_url = "http://127.0.0.1:6060"
        self.server_process = multiprocessing.Process(target=start_dummy_server, daemon=True)
        self.server_process.start()
        # Delay to allow server to spin up
        time.sleep(1.5)

    def store_obs(self, obs):
        self.current_state = obs

    def _get_obs(self):
        return self.current_state

    def _get_info(self):
        return {"current_step": self.current_step}
    
    def _send_action(self, action_idx):
        action = { "move": ["up", "down", "left", "right", "idle"][action_idx] }
        try:
            requests.post(f"{self.api_url}/last-action", json=action, timeout=20)
        except Exception as e:
            print("[ERROR] Failed to send action:", e)

    def _wait_for_new_observation(self):
        start = time.time()
        last_obs = self.current_state

        while time.time() - start < self.timeout:
            try:
                res = requests.get(f"{self.api_url}/observation")
                obs = Parse_data.parse_observation(res.json())
                if obs and obs != last_obs:
                    print(f"recevied state here: {obs}")
                    return np.array(obs, dtype=np.float32)
            except Exception as e:
                print("[WARN] Failed to fetch observation:", e)
            time.sleep(0.1)

        print("[TIMEOUT] Using last known state")
        return np.array(last_obs, dtype=np.float32)

    def reset(self):
        try:
            requests.post(f"{self.api_url}/reset", json={'d':'d'}, timeout=20)
        except Exception as e:
            print("[WARN] Failed to Reset", e)
        self.current_state = self._wait_for_new_observation()
        #np.array([0]*21, dtype=np.float32)
        self.current_step = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self._send_action(action)
        obs = self._wait_for_new_observation()
        #DEbug here
        self.current_state = np.array(obs, dtype=np.float32)
        self.current_step += 1
        reward = 0 #reward function TBD Parse_data.get_reward(obs, actions)
        terminated = self.current_state[3] <= 0  # e.g., player HP
        truncated = self.current_step >= self.max_steps
        return self.current_state, reward, terminated, truncated, self._get_info()

    def close(self):
        if hasattr(self, "server_process") and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join()

# Example usage:
if __name__ == "__main__":
    env = CustomBlankEnv()
    #obs, info = env.reset()
    #print("Initial Observation:", obs)
    for _ in range(10):
        action = np.random.randint(5)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            break

    env.close()