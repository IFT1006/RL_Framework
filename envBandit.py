import numpy as np

from environment import Environnement
from reward import Reward

class EnvBandit(Environnement):
    def __init__(self, agent_count, win_rate, algo):
        if np.argmax(win_rate) != 0:
            raise Exception("Error: index 0 must have the biggest value")
        super().__init__(agent_count, algo)
        self.win_rate = win_rate

    def step(self):
        prev_act = list(self.actions)
        for i in range(self.agent_count):
            action = self.agents[i].train(prev_act[0:i] + prev_act[(i+1):])
            self.actions[i] = action

            # calculate the step reward for the given action
            reward = Reward(self.win_rate, action)
            step_reward = reward.getBanditReward()
            self.agents[i].update(action, step_reward, self.win_rate)