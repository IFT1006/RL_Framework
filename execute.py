import pandas as pd

from environment import Environment
from learningAlgo import LearningAlgo
from agent import Agent

class Execute():
    def __init__(self, iteration, runs, n_agents, win_rate, const, algo):
        # number of iterations of runs
        self.iteration = iteration
        # number of runs in each iteration
        self.runs = runs
        # list of agents
        self.n_agents = n_agents
        self.const = const
        self.algo = algo
        self.win_rate = win_rate

    def getResult(self):
        experiments = []
        for e in range(0, self.iteration):
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

        return { 'avg': pd.DataFrame(experiments).mean(), 'std': pd.DataFrame(experiments).std() } if self.iteration > 1 else { 'agents': agents }