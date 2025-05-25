import numpy as np
from agentSpace import AgentSpace

class Agent:
    def __init__(self, a_space: AgentSpace, algo):

        self.learning_algo = algo
        self.a_space = a_space
        self.cumul_regret = []
        self.cumul_reward = []
        self.reward = []

    def update(self, action, step_reward, win_rate=[]):
        if len(win_rate) == 0  and self.a_space.game == 'Bandit':
            raise Exception("Error: must provide a win rate for the Bandit game!")

        self.a_space.plays[action] += 1
        self.a_space.sums[action] += step_reward
        self.a_space.avg_reward = np.divide(
            self.a_space.sums,
            self.a_space.plays,
            out=np.zeros_like(self.a_space.sums, dtype=float),
            where=self.a_space.plays != 0
        )

        if self.a_space.game == 'Bandit':
            step_regret = max(win_rate) - win_rate[action]

        if self.a_space.game == 'Bandit':
            if self.a_space.t > 1:
                self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
            else:
                self.cumul_regret.append(step_regret)

        if self.a_space.game == 'PD':
            if len(self.cumul_reward) > 1:
                self.cumul_reward.append(self.cumul_reward[-1] + step_reward)
            else:
                self.cumul_reward.append(step_reward)
            self.reward.append(step_reward)

    def train(self, neighbor_actions = []):
        if len(neighbor_actions) != self.a_space.n_neighbors and self.learning_algo.algo_name == 'TUCB':
            raise Exception("Error: neighbor actions must have the same number of neighbors")
        self.a_space.t += 1

        # get action for the current step
        action = self.learning_algo.getAction(neighbor_actions)

        return action
