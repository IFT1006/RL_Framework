from reward import Reward
from environment import Environment

class Agent():
    def __init__(self, env: Environment, algo):
        # average reward
        self.avg_reward = [0]*env.nbr_actions
        # step (play) number
        self.t = 0
        # cumulative regret
        self.cumul_regret = []
        # learning algo to use
        self.env = env
        self.learning_algo = algo

    def update(self, action, step_reward, win_rate):
        step_regret = 0

        for i in range(len(self.env.plays)):
            if action == 0:
                self.env.plays[0] += 1
                self.avg_reward[0] += (step_reward - self.avg_reward[0]) / self.env.plays[0]
                # no added regret since arm 0 is optimal

            elif action == i:
                self.env.plays[i] += 1
                self.avg_reward[i] += (step_reward - self.avg_reward[i]) / self.env.plays[i]

                step_regret = win_rate[0] - win_rate[i]

        if self.t > 1:
            self.cumul_regret.append(self.cumul_regret[-1] + step_regret)
        else:
            self.cumul_regret.append(step_regret)

    def train(self, win_rate, neighbor_actions):
        self.t += 1

        action = self.learning_algo.getAction(self.t, self.avg_reward, neighbor_actions)
        step_reward = Reward(win_rate, action).getReward()
        self.update(action, step_reward, win_rate)
        return action
