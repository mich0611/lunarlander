import gym
import random
import math
import pickle
import numpy.random as rd
from statistics import mean, median
from matplotlib import pyplot as plt

"""
#### System Requirement ####

[Windows]

> pip install gym
> pip install box2d
> pip install pyglet

[macOS]

> pip3 install gym[all]
> pip3 install box2d

If you occur an ImportError like "ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.", please run the following installation.
> pip3 install pyglet==1.5.11
"""

"""
#### Your Job ####
1. Crossover:
        Please implement your crossover operation in the `crossover(cls, parent1, parent2, xover_rate)` function.
        Arguments: parent1 (Chromosome), parent2 (Chromosome), xover_rate (float)
        Return:    two children in a tuple, i.e., (child1, child2); each child is an instance of Chromosome.

2. Mutation:
        Please implement your crossover operation in the `mutate(cls, chrm, mutate_rate)` function.
        Arguments: chrm (Chromosome), mutate_rate (float)
        Return:    None

3. k-tournament selection:
        Please implement the k-tournament selection in the `parent_selection(self, k)` function.
        Arguments: self (GA), k (int)
        Return:    a selected parent (Chromosome)

4. GA parameters:
        Please try different GA parameters and report an appropiate setting for this problem.
"""


#### GA Parameters ####
"""
[YOUR JOB]
Try and find a proper parameter setting.
"""
GA_POPULATION_SIZE = 150
GA_GENERATION      = 200
GA_K_TOURNAMENT    = 10
GA_CROSSOVER_RATE  = 0.7
GA_MUTATION_RATE   = 0.0005

#### Evaluation Parameter ####
"""
This parameter controls the simulation times in one evaluation
Since the simulation environment is randomly generated, the simulation should be run several times to get the reliable performance
Notice: You may lower this parameter to make the whole GA run faster in order to get the sense of how different parameters affect the performance,
but it is suggested that you set it to 20 to get a reliable output for homework assignment
"""
SIMULATIONS_PER_EVALUATION = 15

#### [DON'T CHANGE] Evironment Parameters ####
OBSV_DIM   = 6 # X, Y, V_X, V_Y, ANG, V_ANG 
ACTION_DIM = 2 # MAIN, SIDE
RESOLUTION = 4
N_CASES    = RESOLUTION ** OBSV_DIM
CHRM_DIM   = N_CASES * ACTION_DIM
ACTION_LB  = -1 
ACTION_UB  = 1

env = gym.make("LunarLanderContinuous-v2")

def Tune():
    start = 1
    lst = []
    for i in range(GA_GENERATION):
        lst.append(start)
        start *= 0.9919
    return lst

tune = Tune()

#### [DON'T CHANGE] Please don't change any part of this function. ####
def evaluate(gene:list, repeat=SIMULATIONS_PER_EVALUATION, mode="mean", display=False) -> float:
    rewards = []

    for i in range(repeat):
        env.reset()
        action = [0, 0]
        epsiodic_reward = 0.0
        if display:
            while True:
                env.render()
                obsv, reward, done, _ = env.step(action)
                epsiodic_reward += reward
                action = get_action(obsv, gene)
                if done:
                    break
            rewards.append(epsiodic_reward)
        else:
            while True:
                obsv, reward, done, _ = env.step(action)
                epsiodic_reward += reward
                action = get_action(obsv, gene)
                if done:
                    break
            rewards.append(epsiodic_reward)
    
    if mode == "mean":
        return mean(rewards)

    elif mode == "median":
        return median(rewards)

    elif mode == "min":
        return min(rewards)

    elif mode =="max":
        return max(rewards)

    else:
        raise ValueError("The argument `mode` can be 'mean', 'median', 'min', or 'max', but '{mode}' was given.")


class Chromosome:
    def __init__(self, dim) -> None:
        self.gene    = [random.uniform(ACTION_LB, ACTION_UB) for i in range(dim)]
        self.fitness = 0.0

    @classmethod
    def crossover(cls, parent1, parent2, xover_rate) -> tuple:
        """
        [YOUR JOB]
        The following code implements the uniform crossover.
        Please implement your crossover operator and replace the following code with yours.
        Hint: You may try "n-point crossover", "whole arithmetic crossover", "blend crossover", etc.
        """
        # whole arithmetic crossover

        dim = len(parent1.gene) # length of chromosome
        alpha = 0.7
        child1, child2 = Chromosome(dim), Chromosome(dim)
        
        # xover take places
        if random.random() <= xover_rate: 
            for i in range(dim):
                child1.gene[i] = alpha * parent1.gene[i] + (1-alpha) * parent2.gene[i]
                child2.gene[i] = alpha * parent2.gene[i] + (1-alpha) * parent1.gene[i]
        else:
            child1.gene = parent1.gene
            child2.gene = parent2.gene

        return child1, child2

    @classmethod
    def mutate(cls, chrm, mutate_rate, tune_rate) -> None:
        """
        [YOUR JOB]
        The following code implements the random resetting mutation.
        Please implement your mutation operator and replace the following code with yours.
        Hint: You may try "Gaussian mutation", "creep mutation", etc.
        """
        lower = ACTION_LB
        upper = ACTION_UB
        dim = len(chrm.gene) 
        sigma = 1/4

        for i in range(dim): # gaussian mutation (mean=0, std =0)
            n = random.random()
            if n <= mutate_rate * tune_rate :
                gaussian = random.gauss(0, sigma)
                chrm.gene[i] += gaussian
                if chrm.gene[i] > upper:
                    chrm.gene[i] = upper
                elif chrm.gene[i] < lower:
                    chrm.gene[i] = lower
        return 

class GA:
    def __init__(self, pop_size, k, xover_rate, mutate_rate):
        #### Validate arguments ####
        if (not isinstance(pop_size, int)) or  (pop_size < 1):
            raise ValueError("`pop_size` can only be positive integer.")
        
        if (not isinstance(k, int)) or (k < 1) or (k > pop_size):
            raise ValueError("`k` for tournament selection can only an integer between 1 and `pop_size`.")

        #### Variable declaration ####
        global CHRM_DIM
        self.k   = k
        self.pop = []
        self.pop_size    = pop_size
        self.xover_rate  = xover_rate
        self.mutate_rate = mutate_rate
        self.best_so_far = None
        self.i = 0

        #### Initialize variables ####
        self.pop = [Chromosome(dim=CHRM_DIM) for i in range(pop_size)]
        self.best_so_far = self.pop[0]

        #### Evaluate the initial population ####
        for i, chrm in enumerate(self.pop):
            print(f"Evaluating {(i/self.pop_size)*100: 3.1f}%", end="\r")
            chrm.fitness = evaluate(chrm.gene)

            if chrm.fitness > self.best_so_far.fitness:
                self.best_so_far = chrm
    
    def evolve(self):
        offspring = []

        #### Reproduction ####
        while len(offspring) < self.pop_size:

            #### Parent selection ####
            p1 = self.parent_selection(k=GA_K_TOURNAMENT)
            p2 = self.parent_selection(k=GA_K_TOURNAMENT)

            #### Crossover ####
            c1, c2 = Chromosome.crossover(p1, p2, self.xover_rate)

            #### Mutation ####
            Chromosome.mutate(c1, self.mutate_rate, tune[self.i])
            Chromosome.mutate(c2, self.mutate_rate, tune[self.i])

            offspring += [c1, c2]
        self.i += 1

        #### Evaluate offspring ####
        for i, chrm in enumerate(offspring):
            print(f"Evaluating {(i/self.pop_size)*100: 3.1f}%", end="\r")
            chrm.fitness = evaluate(chrm.gene)

            if chrm.fitness > self.best_so_far.fitness:
                self.best_so_far = chrm

        #### Survival selection ####
        """
        [OPTIONAL]
        You may play with different survival selection strategies.
        I've implemented the (mu +lambda) and (mu, lambda) strategies for you as follows.
        """
        ## (mu + lambda) ##
        intermediate_pop = self.pop + offspring
        self.survival_selection(pool=intermediate_pop, n_survivor=self.pop_size)

        ## (mu, lambda) ##
        # self.survivors(pool=offspring, n_survivor=self.pop_size)   

    def parent_selection(self, k) -> Chromosome:
        """
        [YOUR JOB]
        The following code implements the random parent selection.
        Please implement your parent selection strategy and replace the following code with yours.
        """
        n = random.random()
        inds = [random.randint(0, (self.pop_size-1)) for i in range(k)]
        if n >= 0.1:
            fitnesses = [self.pop[i].fitness for i in inds]
            best = max(fitnesses)
            return self.pop[fitnesses.index(best)]
        else:
            return self.pop[random.choice(inds)]


    def survival_selection(self, pool, n_survivor) -> list:
        """
        I've implement this function for you. No need to change.
        """
        # If `pool` is the offspring only, this performs (mu, lambda) selection.
        # If `pool` is the union of the population and offspring, this performs (mu + lambda) selection.
        
        if n_survivor == len(pool):
            self.pop = pool
        else:
            sorted_pool = sorted(pool, key=lambda chrm:chrm.fitness, reverse=True)
            self.pop    = sorted_pool[:n_survivor]



#### [DON'T CHANGE] Please don't change any part of this function. ####
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

####################################################


#### Initialize a GA instance ####
ga = GA(
    pop_size=GA_POPULATION_SIZE,
    k=GA_K_TOURNAMENT,
    xover_rate=GA_CROSSOVER_RATE,
    mutate_rate=GA_MUTATION_RATE,
)

score = []
print(f"Generation {0: 4d}, best fitness = {ga.best_so_far.fitness:4.2f}")
evaluate(ga.best_so_far.gene, display=True)
score.append(ga.best_so_far.fitness)

#### Evolve until termination criterion is met. #####
for i in range(GA_GENERATION):
    ga.evolve()
    print(f"Generation {(i+1): 4d}, best fitness = {ga.best_so_far.fitness:4.2f}")
    score.append(ga.best_so_far.fitness)
    evaluate(ga.best_so_far.gene, display=True)

#### Store the best solution ####
with open('best_gene.pickle', 'wb') as f:
        pickle.dump(ga.best_so_far.gene, f)
# LunarLander_HW.py
# 目前顯示的是「LunarLander_HW.py」。

with open('data.txt', 'w') as f:
    for i in score:
        f.write(str(i))
        f.write('\n')

def draw():
    plt.figure(figsize=(8,6), dpi = 100)
    plt.plot(score)
    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.title('Any time behavior')
    plt.savefig('Any time behavior.png')
draw()
