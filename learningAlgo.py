import numpy as np
import random

from agentSpace import AgentSpace

class LearningAlgo:
    def __init__(self, constant, algo_name, a_space: AgentSpace):
        # constant C
        self.constant = constant
        # algorithm name
        self.algo_name = algo_name
        self.a_space = a_space
        self.init_iteration = 0

    def getInitialState(self):
        first_time = False
        action = 0
        if self.a_space.game == "Bandit":
            for i in range(self.a_space.n_arms):
                # play arm for the first time
                if self.a_space.plays[i] == 0:
                    action = i
                    first_time = True
                    break
        elif self.a_space.game == "PD":
            # A-A-B-B
            if self.a_space.a_id == 1:
                for i in range(self.a_space.n_arms):
                    if self.a_space.plays[i] < self.a_space.n_arms:
                        action = i
                        first_time = True
                        break
            # A-B-A-B
            elif self.a_space.a_id == 2 and self.a_space.t <= self.a_space.n_arms**2:
                for i in range(self.a_space.n_arms):
                    if self.a_space.plays[i] == self.init_iteration:
                        action = i
                        first_time = True
                        if i == self.a_space.n_arms - 1:
                            self.init_iteration += 1
                        break

        return {'action': action, 'first_time': first_time }

    def getTUCBAction(self, neighbor_actions, first_time, action):
        # calculate the target play once t > 1
        if self.a_space.t > 1:
            for a in neighbor_actions:
                self.a_space.target_plays[a] += 1 / self.a_space.n_neighbors

        if not first_time:            
            # start the algo after initialization
            action_val = np.zeros(self.a_space.n_arms)
            est_opt = np.sqrt(self.constant * np.log(self.a_space.t) / self.a_space.plays)
            target_opt = np.sqrt(((self.a_space.target_plays - self.a_space.plays) / self.a_space.target_plays).clip(min=0))
            action_val = self.a_space.avg_reward + est_opt * target_opt

            best = np.flatnonzero(np.isclose(action_val, action_val.max()))
            action = np.random.choice(best)
        return action

    def getUCBAction(self, first_time, action):
        if not first_time:

            # start the algo after initialization
            action_val = np.zeros(self.a_space.n_arms)
            est_opt = np.sqrt(self.constant * np.log(self.a_space.t) / self.a_space.plays)
            action_val = self.a_space.avg_reward + est_opt

            best = np.flatnonzero(np.isclose(action_val, action_val.max()))
            action = np.random.choice(best)
        return action

    def getAction(self, neighbor_actions):
        res = self.getInitialState()
        first_time = res['first_time']
        action = res['action']

        match self.algo_name:
            case "TUCB":
                return self.getTUCBAction(neighbor_actions, first_time, action)
            case "UCB":
                return self.getUCBAction(first_time, action)
            case _:
                return None