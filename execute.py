import pandas as pd
import numpy as np

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from envPD import EnvPD
from envBandit import EnvBandit

class Execute:
    def __init__(self, instance, runs, n_agents, const, algo):
        # number of instances of runs
        self.instance = instance
        # number of runs in each instance
        self.runs = runs
        # list of agents
        self.n_agents = n_agents
        self.const = const
        self.algo = algo

    def getPDResult(self):
        experiments = []
        for e in range(0, self.instance):
            A_PD = np.array([[3, 0],
                             [5, 1]]).astype(float)
            B_PD = np.array([[3, 5],
                             [0, 1]]).astype(float)
            matrices = [A_PD, B_PD]
            env = EnvPD(matrices, 2)
            plays = []

            # print to trace the progress
            print(e)

            for j in range(0, self.n_agents):
                a_space = AgentSpace(2, self.n_agents, 'PD')
                learning_algo = LearningAlgo(self.const, self.algo, a_space)
                env.ajouter_agents(Agent(a_space, learning_algo))

            for t in range(0, self.runs):
                actions = env.step()
                plays.append(actions['a2'])

            experiments.append(plays)
        return {'prop': pd.DataFrame(experiments).mean()}

    def getBanditResult(self, win_rate, use_rand_win):
        experiments = []
        for e in range(0, self.instance):
            # define the random win rate for the current instance if asked - index 0 should have the biggest value
            if use_rand_win is True:
                rand_win_rate = np.random.rand(2)
                while rand_win_rate[0] <= rand_win_rate[1]:
                    rand_win_rate = np.random.rand(2)
            # print to trace the progress
            print(e)
            env = EnvBandit(self.n_agents, win_rate if use_rand_win is False else rand_win_rate)

            for i in range(0, self.n_agents):
                a_space = AgentSpace(len(win_rate), self.n_agents, 'Bandit')
                learning_algo = LearningAlgo(self.const, self.algo, a_space)
                env.ajouter_agents(Agent(a_space, learning_algo))

            for t in range(0, self.runs):
                env.step()

            experiments.append(env.agents[0].cumul_regret)

        # if more than 1 instance the key "agents" returns the cumul_regret of agents in the last instance
        return { 'avg': pd.DataFrame(experiments).mean(), 'std': pd.DataFrame(experiments).std(), 'agents': env.agents }