import gym
import numpy as np
import time
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n
ACTIONS = env.action_space.n
### CONSTANTS
#no. of run in the environment
EPISODES = 10000
# max steps for each run
MAX_STEPS = 100
LEARNING_RATE = 0.81
GAMMA = 0.96
RENDER = False
# for action randomness
epsilon = 0.9

Q = np.zeros((STATES, ACTIONS))

if 1==0:
    print('No of states: '+str(STATES)) # states
    print('No. of actions: '+str(ACTIONS)) # per each state
    env.reset()
    # get a random action
    action = env.action_space.sample()
    # returns some info for taking that action (4 values)
    observation, reward, done, info = env.step(action)
    print('Action: '+str(action),
          '\nObservation: '+str(observation),
          '\nReward: '+str(reward),
          '\nStatus: '+str(done),
          '\nInfo: '+str(info))

    # S-start; F-frozen; H-hole; G-goal
    env.render()

# Pick action

rewards = []
for episode in range(EPISODES):
    state = env.reset()
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        next_state, reward, done, _ = env.step(action)
        Q[state, action] = Q[state, action] + LEARNING_RATE*(reward + GAMMA*np.max(Q[next_state, :]) - Q[state, action])
        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break
print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}")

def get_average(values):
    return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i + 100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()

