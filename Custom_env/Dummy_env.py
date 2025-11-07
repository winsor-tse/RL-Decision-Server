import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from Parse_Data.Parse_data import parse_observation
import time
import requests

DIRECTION_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "idle": 4  # if needed
}

ACTIONS = ['up', 'down', 'left', 'right', 'attack']

"""
JS code here:
try {
        const obsRes = await fetch("http://localhost:8000/observation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(worldState)
        });

        const obsResult = await obRes.json();
        console.log("Observation sent:", obsResult);


        // 2. Get the last action taken (you can also combine this into one POST if preferred)
        const actionRes = await fetch("http://localhost:8000/last-action", {
            method: "GET"
        });

        const actionData = await actionRes.json();
        console.log("Received action:", actionData);

        if (actionData.move) {
            movePlayerDirection(actionData.move); // "left", "right", etc.
        }

         const resetReset = await fetch("http://localhost:8000/reset", {
            method: "GET"
        });

        const resetData = await Reset.json();
        //Not sure set Player Hp to -1 for death
        if (resetData){
            getNetwork().playerHp(-1);
        }

    } catch (err) {
        console.error("AI server error:", err);
    }


"""

class CustomBlankEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.Actions = ACTIONS
        self.single_action_space = spaces.Discrete(len(self.Actions))
        self.single_observation_space = spaces.Box(low=0, high=math.inf, shape=(21,), dtype=np.float32)
        self.current_state = np.zeros(4, dtype=np.float32)
        self.current_step = 0
        self.max_steps = 100
        self.api_url = 'http://localhost:8000/'
        self.timeout = 1000

    def store_obs(self, obs):
        self.current_state = obs

    def _get_obs(self):
        return self.current_state

    def _get_info(self):
        return {"current_step": self.current_step}
    
    def _send_action(self, action_idx):
        action = { "move": ["up", "down", "left", "right", "idle"][action_idx] }
        try:
            requests.post(f"{self.api_url}/last-action", json=action, timeout=2)
        except Exception as e:
            print("[ERROR] Failed to send action:", e)

    def _wait_for_new_observation(self):
        start = time.time()
        last_obs = self.current_state

        while time.time() - start < self.timeout:
            try:
                res = requests.get(f"{self.api_url}/observation")
                obs = parse_observation(res.json())
                if obs and obs != last_obs:
                    return np.array(obs, dtype=np.float32)
            except Exception as e:
                print("[WARN] Failed to fetch observation:", e)
            time.sleep(0.1)

        print("[TIMEOUT] Using last known state")
        return np.array(last_obs, dtype=np.float32)

    def reset(self):
        try:
            requests.post(f"{self.api_url}/reset", json={'d':'d'}, timeout=2)
        except Exception as e:
            print("[WARN] Failed to Reset", e)
        self.current_state = np.array([0]*21, dtype=np.float32)
        self.current_step = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self._send_action(action)
        obs = self._wait_for_new_observation()
        self.current_state = np.array(obs, dtype=np.float32)
        self.current_step += 1
        reward = 0 #reward function TBD
        terminated = self.current_state[3] <= 0  # e.g., player HP
        truncated = self.current_step >= self.max_steps
        return self.current_state, reward, terminated, truncated, self._get_info()


# Example usage:
if __name__ == "__main__":
    env = CustomBlankEnv()
    obs, info = env.reset()
    print("Initial Observation:", obs)

    for _ in range(10):
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            break

    env.close()