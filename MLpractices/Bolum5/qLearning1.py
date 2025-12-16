import gym
import numpy as np

environment = gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

environment.reset()

nStates = environment.observation_space.n
nActions = environment.action_space.n
qTable = np.zeros((nStates, nActions))

print('Q-table:')
print(qTable)

action = environment.action_space.sample()
"""
sol: 0
assagi: 1
sag: 2
yukari: 3
"""

# Yenidurum, odul, bitis, yapildi
newState, reward, done, info, _ = environment.step(action)

# %%

import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

environment = gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

environment.reset()

nStates = environment.observation_space.n
nActions = environment.action_space.n
qTable = np.zeros((nStates, nActions))

print('Q-table:')
print(qTable)

episodes = 1000 # episode
alpha = 0.5 # learning rate
gamma = 0.9 # discount rate
outcomes = []

# training 

for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False # agent's success state
    
    outcomes.append('Failure')
    
    while not done: # state will continue until agent succeeds
        
        # action
        if np.max(qTable[state]) >0:
            action = np.argmax(qTable[state])
            
        else:
            action = environment.action_space.sample()

        newState, reward, done, info, _ = environment.step(action)
        
        # update qTable 
        
        malakia = reward+gamma*np.max(qTable[newState]) -qTable[state,action]
        
        qTable[state, action] = qTable[state, action] + alpha *malakia
        
        state = newState
        
        if reward:
            outcomes[-1] = 'Success'

print('qTable After Training')
print(qTable)

plt.bar(range(episodes),outcomes)

# test

epique = 100
nSuc = 0

for _ in tqdm(range(epique)):
    state, _ = environment.reset()
    done = False
    
    while not done:
        
        if np.max(qTable[state]):
            action = np.argmax(qTable[state])
            
        else:
            action = environment.action_space.sample()
            
        newState, reward, done, info, _ = environment.step(action)
        
        state = newState

        nSuc += reward
    
print('Number of Success: ', 100*nSuc/epique)

























