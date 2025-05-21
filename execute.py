import pandas as pd
import numpy as np

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from environment import Environnement

class Execute:
    def __init__(self, instance, runs, n_agents, win_rate, const, algo, use_rand_win):
        # number of instances of runs
        self.instance = instance
        # number of runs in each instance
        self.runs = runs
        # list of agents
        self.n_agents = n_agents
        self.const = const
        self.algo = algo
        self.win_rate = win_rate
        self.use_rand_win = use_rand_win

    # def getPDResult(self):
    #     A_PD = np.array([[3, 0],
    #                      [5, 1]]).astype(float)
    #     B_PD = np.array([[3, 5],
    #                      [0, 1]]).astype(float)
    #     matrices = [A_PD, B_PD]
    #     experiments = []
    #     for i in range(0, self.instance):
    #         print(i)
    #         env = Environnement(matrices, 2)
    #
    #         for j in range(0, self.n_agents):
    #             a_space = AgentSpace(2, self.n_agents - 1, self.n_agents, 'PD')
    #             learning_algo = LearningAlgo(self.const, self.algo, a_space)
    #             env.ajouter_agents(Agent(a_space, learning_algo))
    #
    #         for t in range(0, self.runs):
    #             agents[0].train(self.win_rate, [])
    #             agents[1].train(self.win_rate, [])
    #
    #         experiments.append(agents[0].cumul_regret)

    def getBanditResult(self):
        experiments = []
        for e in range(0, self.instance):
            # define the random win rate for the current instance if asked - index 0 should have the biggest value
            if self.use_rand_win is True:
                self.win_rate = np.random.rand(2)
                while self.win_rate[0] <= self.win_rate[1]:
                    self.win_rate = np.random.rand(2)

            # initialize the list of agents and actions for the current iteration
            agents = []
            actions = []
            # print to terminal to track the progress
            print(e)

            for i in range(0, self.n_agents):
                a_space = AgentSpace(len(self.win_rate), self.n_agents - 1, self.n_agents, 'Bandit')
                learning_algo = LearningAlgo(self.const, self.algo, a_space)
                agents.append(Agent(a_space, learning_algo))
                actions.append(0)

            for t in range(0, self.runs):
                # copy list of previous action
                prev_act = list(actions)

                for a in range(len(agents)):
                    actions[a] = agents[a].train(self.win_rate, prev_act[0:a] + prev_act[(a + 1):])

            experiments.append(agents[0].cumul_regret)

        # if more than 1 instance the key "agents" returns the cumul_regret of agents in the last instance
        return { 'avg': pd.DataFrame(experiments).mean(), 'std': pd.DataFrame(experiments).std(), 'agents': agents }