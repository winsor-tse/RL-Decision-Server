#TODO: Use Past data from Off policy algorithms like DQN - DDPD/SAC with Experience Replay buffer to teach 
"""

I want a replay function that stores only top trajectories. (Ill add that in DQN server most likely)

Store that into a storage space (This will act as the teacher forcing for imatation learning)

It can also act as a replay viewer similar to videos. (This is not pygame we can not record screen but we have trajectories)

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