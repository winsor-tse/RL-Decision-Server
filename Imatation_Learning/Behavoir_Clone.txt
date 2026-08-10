#TODO: Use Past data from Off policy algorithms like DQN - DDPD/SAC with Experience Replay buffer to teach 
"""
In online RL, we send payloads to take actions.

In offline RL/ Behavoir cloning. I need to record what the player is doing in the parsed State and action.

All we are going to do is build a CLI that message = self.socket.recv_json().

No need to send an action back. The idea is to prompt on CLI when to record and record it into a SQL Lite Database.

YOu can then custom parse obs based on something like running dataset from Env_16  parse_observation.

This is going to the the expert trajectories to be intializing the policy with.

Store and train/intialize a policy doing this.

"""

"""
Advanced Option:

Set Up HRL or Goal conditioned RL.

BC can intialise policies such as attack/defend/retreat from recorded traj.

One environment step is a *high-level* step: the high level policy assigns a low-level
    policy (Attack, Defend, or Retreat) for the duration of steps, and that policy is rolled out for
    `hl_sim_steps` low-level steps. Rewards accumulate over the inner steps.

(Attack, Defend, or Retreat).Low-Level GCRL Policy: Translates that mode into continuous executable actions by treating the mode's destination (e.g., enemy coordinates, defensive cover, or extraction zones) as the conditioned input goal g.

"""


"""
Collect expert or high-return transitions from DQN replay buffer.
Build a dataset:
X = states
y = actions
Train the Q-network or policy head with classification loss.
Save the behavior-cloned model.
Use that model as initialization for DQN or even for critic in PPO.
"""


