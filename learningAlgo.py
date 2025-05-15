import numpy as np
import random

from environment import Environment

class LearningAlgo:
    def __init__(self, constant, algo_name, env: Environment):
        # constant C
        self.constant = constant
        # estimation optimism for each action
        self.est_opt = [0]*env.nbr_actions
        # target optimism for each action
        self.target_opt = [0]*env.nbr_actions
        # target optimism for each action
        self.action_val = [0] * env.nbr_actions
        # algorithm name
        self.algo_name = algo_name
        self.env = env

    def getTUCBAction(self, t, avg_reward, neighbor_actions):
        '''Returns the action'''
        action = 0
        if t > 1:
            for a in neighbor_actions:
                self.env.target_plays[a] += 1 / self.env.nbr_neighbors

        first_time = False
        for i in range(self.env.nbr_actions):
            if self.env.plays[i] == 0:
                # play arm for the first time
                action = i
                first_time = True
                break

        if not first_time:
            for i in range(self.env.nbr_actions):
                self.est_opt[i] = np.sqrt(self.constant * np.log(t) / self.env.plays[i])
                self.target_opt[i] = np.sqrt((self.env.target_plays[i] - self.env.plays[i]) / self.env.target_plays[i]) if (
                            (self.env.target_plays[i] - self.env.plays[i]) > 0) else 0

            for i in range(self.env.nbr_actions):
                self.action_val[i] = avg_reward[i] + self.est_opt[i] * self.target_opt[i]

            if len(set(self.action_val)) > 1:
                action = np.argmax(self.action_val)
            else:
                # tie breaker
                action = random.choice([0, self.env.nbr_actions - 1])
        return action

    def getAction(self, t, avg_reward, neighbor_actions):
        if self.algo_name == "TUCB":
            return self.getTUCBAction(t, avg_reward, neighbor_actions)
        return None