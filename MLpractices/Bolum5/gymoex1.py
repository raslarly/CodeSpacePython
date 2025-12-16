# this code is not working due to some sort of uninstallable library ?

import gym

# Initialise the environment
env = gym.make('LunarLander',render_mode='human')

# reset the environment to generate the first observation

observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    
    action = env.action_space.sample()
    
    # step() (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated
    # or truncated 
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info =     env.reset()

env.close()





























