import gym
import numpy as np
import random
from tqdm import tqdm

env = gym.make('Taxi', render_mode = 'ansi')
env.reset()

print(env.render())

"""
0: guney
1: kuzey
2: dogu
3: bati
4: yolcuyu al
5: yolcuyu indir
"""

aSpace = env.action_space.n
sSpace = env.observation_space.n

qTable = np.zeros((sSpace, aSpace))

alpa = 0.1 # Learning rate
gama = 0.6 # Discount rate
expl = 0.1 # epsilon

for i in tqdm(range(1,100001)):
    state, _ = env.reset()
    
    done = False
    
    while not done:
        
        if random.uniform(0,1) < expl: # explore rate %10
            env.action_space.sample()
        else: # exploit
            action = np.argmax(qTable[state])

        newState, reward, done, info, _ = env.step(action)
        
        maku = reward + gama * np.max(qTable[newState]) - qTable[state, action]
        
        qTable[state, action] = qTable[state, action] + alpa * maku

        state = newState

print('Training finished')

# test

totEpo, totPen = 0, 0
episodes = 100

for i in tqdm(range(episodes)):
    
    state, _ = env.reset()

    epo,pen,reward = 0,0,0
    
    done = False
    
    while not done:
        
        action = np.argmax(qTable[state])
        
        newState, reward, done, info, _ = env.step(action)
        
        state = newState
        
        if reward == -10:
            pen += 1
        
        epo += 1
        
    totEpo += epo
    totPen += pen

print('result after {} episodes.'.format(episodes))
print('Average timesteps per episode: ',totEpo/episodes)
print('Average penalties: ',totPen/episodes)




















