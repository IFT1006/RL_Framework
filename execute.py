import pandas as pd
import numpy as np

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from envPD import EnvPD
from envBandit import EnvBandit

class Execute:
    def __init__(self, instance, runs, n_agents, const):
        # number of instances of runs
        self.instance = instance
        # number of runs in each instance
        self.runs = runs
        # list of agents
        self.n_agents = n_agents
        self.const = const

    def getPDResult(self, matrices, algo):
        experiments = []
        for e in range(0, self.instance):
            env = EnvPD(matrices, self.n_agents)
            plays = []

            # print to trace the progress
            print(e)

            for j in range(0, self.n_agents):
                a_space = AgentSpace(len(matrices[0]), self.n_agents, 'PD', j+1)
                learning_algo = LearningAlgo(self.const, algo[j], a_space)
                env.ajouter_agents(Agent(a_space, learning_algo))

            for t in range(0, self.runs):
                actions = env.step()
                plays.append(actions['a2'])

            experiments.append(plays)

        prop = pd.DataFrame(experiments).mean() if len(matrices[0]) == 2 else (
            pd.DataFrame(experiments).apply(lambda col: col.value_counts(normalize=True)).fillna(0).sort_index())
        return {'prop': prop }

    def getBanditResult(self, win_rate, use_rand_win, algo):
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
                a_space = AgentSpace(len(win_rate), self.n_agents, 'Bandit', i+1)
                learning_algo = LearningAlgo(self.const, algo, a_space)
                env.ajouter_agents(Agent(a_space, learning_algo))

            for t in range(0, self.runs):
                env.step()

            experiments.append(env.agents[0].cumul_regret)

        # if more than 1 instance the key "agents" returns the cumul_regret of agents in the last instance
        return { 'avg': pd.DataFrame(experiments).mean(), 'std': pd.DataFrame(experiments).std(), 'agents': env.agents }