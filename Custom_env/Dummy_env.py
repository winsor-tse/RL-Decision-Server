import gymnasium as gym
from gymnasium import spaces
import numpy as np


"""
TODO: Fast API's hosting of Server

Server start up will run the app, listening for states

Step will receive state (parsed into digestable format) and respond with an action

getreward() -> function will determine reward based on game heuristics

"""


class CustomBlankEnv(gym.Env):
    def __init__(self):
        super().__init__()
        #TODO: Can make actions configurable
        self.Actions = ('up','down','left','right','attack')
        self.single_action_space = spaces.Discrete(len(self.Actions))
        self.single_observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.current_state = np.zeros(4, dtype=np.float32)
        self.current_step = 0
        self.max_steps = 100


    def _get_obs(self):
        # This method should return the current observation based on the environment's state
        return self.current_state

    def _get_info(self):
        # This method should return any auxiliary information (e.g., performance metrics)
        return {"current_step": self.current_step}

    def reset(self, seed=None, options=None):
        #super().reset(seed=seed) # Important for seeding the environment's PRNG
        # Reset environment to an initial state
        # Reset intial state can just be np.zeros(State Space)
        self.current_state = self.np_random.uniform(low=0, high=1, size=(4,)).astype(np.float32)
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        self.current_state += self.np_random.uniform(low=-0.1, high=0.1, size=(4,)).astype(np.float32)
        self.current_state = np.clip(self.current_state, 0, 1) # Keep state within bounds
        reward = -0.01
        terminated = False
        truncated = self.current_step >= self.max_steps
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    #Useless?
    def render(self):
        # Implement rendering logic here (e.g., print state, visualize with Matplotlib)
        if self.render_mode == "human":
            print(f"Current State: {self.current_state}, Step: {self.current_step}")
        elif self.render_mode == "rgb_array":
            # Return an array representing the visual state (e.g., a simple image)
            # This is a placeholder; a real implementation would generate a meaningful image
            return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        # Clean up any resources (e.g., close rendering windows)
        pass

# Example usage:
if __name__ == "__main__":
    env = CustomBlankEnv()
    obs, info = env.reset(seed=42)
    print("Initial Observation:", obs)

    for _ in range(10):
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            break

    env.close()