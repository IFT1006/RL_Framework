import pandas as pd
import numpy as np
from tqdm import tqdm

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from envPD import EnvPD
from envBandit import EnvBandit

class Execute:
    def __init__(self, n_instance, n_runs, n_agents, const):
        self.n_instance = n_instance
        self.n_runs = n_runs
        self.n_agents = n_agents
        self.const = const

    def runOnePDExperiment(self, matrices, algo, noise_dist='uniform', noise_params=(0.0, 0.05)):

        # Initialisation de l'environnement
        env = EnvPD(matrices, self.n_agents, noise_dist, noise_params)
        for agent in range(0, self.n_agents):
            a_space = AgentSpace(len(matrices[0]), self.n_agents, 'PD', agent+1)
            learning_algo = LearningAlgo(self.const, algo[agent], a_space)
            env.ajouter_agents(Agent(a_space, learning_algo))
        
        plays = []
        for time_step in range(0, self.n_runs):
            actions = env.step()
            plays.append([actions[f'a{k+1}'] for k in range(self.n_agents)])

        cumul_rewards = [env.agents[k].cumul_reward for k in range(self.n_agents)]
        rewards = [env.agents[k].reward for k in range(self.n_agents)]

        return plays, cumul_rewards, rewards
    

    def getPDResult(self, matrices, algo, noise_dist='uniform', noise_params=(0.0, 0.05)):

        # Boucle sur les itérations
        experiments = []
        experiments_rewards_cumul = []
        experiments_rewards = []
        for _ in tqdm(range(0, self.n_instance)):
            plays, cumul_rewards, rewards = self.runOnePDExperiment(matrices, algo, noise_dist, noise_params)
            experiments.append(plays)
            experiments_rewards_cumul.append(cumul_rewards)
            experiments_rewards.append(rewards)

        # Enregistrement des résultats dans un Dataframe à faire

        return {'plays': plays, 
                'experiments_rewards_cumul': np.mean(np.array(experiments_rewards_cumul),axis=0), 
                'experiments_rewards': np.mean(np.array(experiments_rewards),axis=0)}

    def getBanditResult(self, win_rate, use_rand_win, algo):

        # Julien: probalement à revoir, j'ai fait beaucoup de changements.

        experiments = []
        for e in tqdm(range(0, self.n_instance)):
            # define the random win rate for the current instance if asked - index 0 should have the biggest value
            if use_rand_win is True:
                rand_win_rate = np.random.rand(2)
                while rand_win_rate[0] <= rand_win_rate[1]:
                    rand_win_rate = np.random.rand(2)

            env = EnvBandit(self.n_agents, win_rate if use_rand_win is False else rand_win_rate)

            for i in range(0, self.n_agents):
                a_space = AgentSpace(len(win_rate), self.n_agents, 'Bandit', i+1)
                learning_algo = LearningAlgo(self.const, algo, a_space)
                env.ajouter_agents(Agent(a_space, learning_algo))

            for t in range(0, self.n_runs):
                env.step()

            experiments.append(env.agents[0].cumul_regret)

        # if more than 1 instance the key "agents" returns the cumul_regret of agents in the last instance
        return { 'avg': pd.DataFrame(experiments).mean(), 'std': pd.DataFrame(experiments).std(), 'agents': env.agents }
