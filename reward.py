import numpy as np

class Reward():
    def __init__(self, win_rate, arm_played):
        # win rate
        self.win_rate = win_rate
        # arm played
        self.arm_played = arm_played

    def getBanditReward(self):
        if len(self.win_rate) == 0:
            raise Exception("Error: win rate must not be empty for Bandit game")
        pull = np.random.rand()

        for i in range(0, len(self.win_rate)):
            if self.arm_played == i and pull < self.win_rate[i]:
                # win
                return 1
        # loss
        return 0