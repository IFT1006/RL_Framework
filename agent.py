import numpy as np
from reward import Reward
from agentSpace import AgentSpace

class Agent():
    def __init__(self, a_space: AgentSpace, algo):
        # cumulative regret
        self.cumul_regret = []
        # learning algo to use
        self.learning_algo = algo
        self.a_space = a_space

    def update(self, action, step_reward, win_rate):
        step_regret = 0

        if action == 0:
            self.a_space.plays[0] += 1
            self.a_space.avg_reward[0] += (step_reward - self.a_space.avg_reward[0]) / self.a_space.plays[0]
            # no added regret since arm 0 is optimal
        else:
            for i in range(1, self.a_space.n_arms):
                self.a_space.plays[i] += 1
                self.a_space.avg_reward[i] += (step_reward - self.a_space.avg_reward[i]) / self.a_space.plays[i]

                step_regret = win_rate[0] - win_rate[i]
        if self.a_space.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

    def train(self, win_rate, neighbor_actions):
        if np.argmax(win_rate) != 0:
            raise Exception("Error: index 0 must have the biggest value")
        if len(neighbor_actions) != self.a_space.n_neighbors and self.a_space.game == 'Bandit':
            raise Exception("Error: neighbor actions must have the same number of neighbors")
        self.a_space.t += 1

        # get action for the current step
        action = self.learning_algo.getAction(neighbor_actions)

        # calculate the step reward for the given action
        reward = Reward(win_rate, action)
        step_reward = reward.getBanditReward() if self.a_space.game == 'Bandit' else reward.getPDReward()

        self.update(action, step_reward, win_rate)

        return action
