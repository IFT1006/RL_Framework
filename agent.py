import numpy as np
from reward import Reward
from environment import Environment

class Agent():
    def __init__(self, env: Environment, algo):
        # cumulative regret
        self.cumul_regret = []
        # learning algo to use
        self.learning_algo = algo
        self.env = env

    def update(self, action, step_reward, win_rate):
        step_regret = 0

        if action == 0:
            self.env.plays[0] += 1
            self.env.avg_reward[0] += (step_reward - self.env.avg_reward[0]) / self.env.plays[0]
            # no added regret since arm 0 is optimal
        else:
            for i in range(1, self.env.n_arms):
                self.env.plays[i] += 1
                self.env.avg_reward[i] += (step_reward - self.env.avg_reward[i]) / self.env.plays[i]

                step_regret = win_rate[0] - win_rate[i]

        if self.env.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

    def train(self, win_rate, neighbor_actions):
        if np.argmax(win_rate) != 0:
            raise Exception("Error: index 0 must have the biggest value")
        self.env.t += 1

        # get action for the current step
        action = self.learning_algo.getAction(neighbor_actions)

        # calculate the step reward for the given action
        step_reward = Reward(win_rate, action).getReward()

        self.update(action, step_reward, win_rate)

        return action
