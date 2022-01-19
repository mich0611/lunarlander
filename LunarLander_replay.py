import gym
import pickle
import math
from statistics import mean

env = gym.make("LunarLanderContinuous-v2")

def evaluate(gene:list, repeat=10, display=False) -> float:
    rewards = []

    for i in range(repeat):
        env.reset()
        action = [0, 0]
        episodic_reward = 0
        if display:
            while True:
                env.render()
                obsv, reward, done, _ = env.step(action)
                episodic_reward += reward
                action = get_action(obsv, gene)
                if done:
                    break
            rewards.append(episodic_reward)
        else:
            while True:
                obsv, reward, done, _ = env.step(action)
                episodic_reward += reward
                action = get_action(obsv, gene)
                if done:
                    break
            rewards.append(episodic_reward)
    
    return mean(rewards)

def get_action(observation:list, gene:list) -> list:
    obsv_dim  = 6
    obsv_grid = [ 
        [0.5,  0.0, -0.5],      # x position
        [0.7,  0.1, -0.5],      # x volecity
        [0.5,  0.0, -0.5],      # y position
        [0.0, -0.5, -1.0],      # y volecity
        [1.0,  0.0, -1.0],      # angle
        [2.0,  0.0, -2.0],      # angular volecity
    ]

    obsv  = observation[:obsv_dim]
    level = [None] * obsv_dim

    for i in range(obsv_dim):
        if obsv[i] > obsv_grid[i][0]:
            level[i] = 0
        elif obsv[i] > obsv_grid[i][1]:
            level[i] = 1
        elif obsv[i] > obsv_grid[i][2]:
            level[i] = 2
        else:
            level[i] = 3  
    
    policy_i = 0
    for i in range(obsv_dim):
        policy_i += level[i] * math.pow(4, i)
    
    policy_i = int(policy_i) * 2
    action   = [gene[policy_i], gene[policy_i+1]]
    
    return action

with open('best_gene.pickle', 'rb') as f:
    gene = pickle.load(f)
    evaluate(gene, repeat=20, display=True)