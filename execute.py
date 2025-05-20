import pandas as pd
import numpy as np

from environment import Environment
from learningAlgo import LearningAlgo
from agent import Agent

class Execute:
    def __init__(self, instance, runs, n_agents, win_rate, const, algo, use_rand_win):
        # number of iterations of runs
        self.instance = instance
        # number of runs in each iteration
        self.runs = runs
        # list of agents
        self.n_agents = n_agents
        self.const = const
        self.algo = algo
        self.win_rate = win_rate
        self.use_rand_win = use_rand_win

    def getResult(self):
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
                env = Environment(len(self.win_rate), self.n_agents - 1, self.n_agents)
                learning_algo = LearningAlgo(self.const, self.algo, env)
                agents.append(Agent(env, learning_algo))
                actions.append(0)

            for t in range(0, self.runs):
                # copy list of previous action
                prev_act = list(actions)

                for a in range(len(agents)):
                    actions[a] = agents[a].train(self.win_rate, prev_act[0:a] + prev_act[(a + 1):])

            experiments.append(agents[0].cumul_regret)

        return { 'avg': pd.DataFrame(experiments).mean(), 'std': pd.DataFrame(experiments).std() } if self.instance > 1 else { 'agents': agents }