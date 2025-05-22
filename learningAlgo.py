import numpy as np
import random

from agentSpace import AgentSpace

class LearningAlgo:
    def __init__(self, constant, algo_name, a_space: AgentSpace):
        # constant C
        self.constant = constant
        # estimation optimism for each action
        self.est_opt = [0] * a_space.n_arms
        # target optimism for each action
        self.target_opt = [0]*a_space.n_arms
        # action value for each action
        self.action_val = [0] * a_space.n_arms
        # algorithm name
        self.algo_name = algo_name
        self.a_space = a_space

    def getTUCBAction(self, neighbor_actions):
        # calculate the target play once t > 1
        if self.a_space.t > 1:
            for a in neighbor_actions:
                self.a_space.target_plays[a] += 1 / self.a_space.n_neighbors

        first_time = False
        for i in range(self.a_space.n_arms):
            # play arm for the first time
            if self.a_space.plays[i] == 0:
                action = i
                first_time = True
                break

        if not first_time:
            # start the algo once each action is played once
            for i in range(self.a_space.n_arms):
                self.est_opt[i] = np.sqrt(self.constant * np.log(self.a_space.t) / self.a_space.plays[i])
                self.target_opt[i] = np.sqrt((self.a_space.target_plays[i] - self.a_space.plays[i]) / self.a_space.target_plays[i]) if (
                            (self.a_space.target_plays[i] - self.a_space.plays[i]) > 0) else 0
                self.action_val[i] = self.a_space.avg_reward[i] + self.est_opt[i] * self.target_opt[i]

            if len(set(self.action_val)) > 1:
                action = np.argmax(self.action_val)
            else:
                # tie breaker
                action = random.choice([0, self.a_space.n_arms - 1])
        return action

    def getUCBAction(self):
        first_time = False
        #TODO: initialisation (boucle de 4 pour initialiser AA AB ... etc.)
        for i in range(self.a_space.n_arms):
            # play arm for the first time
            if self.a_space.plays[i] == 0:
                action = i
                first_time = True
                break

        if not first_time:
            # start the algo once each action is played once
            for i in range(self.a_space.n_arms):
                self.est_opt[i] = np.sqrt(self.constant * np.log(self.a_space.t) / self.a_space.plays[i])
                self.action_val[i] = self.a_space.avg_reward[i] + self.est_opt[i]

            if len(set(self.action_val)) > 1:
                action = np.argmax(self.action_val)
            else:
                # tie breaker
                action = random.choice([0, self.a_space.n_arms - 1])
        return action

    def getAction(self, neighbor_actions):
        match self.algo_name:
            case "TUCB":
                return self.getTUCBAction(neighbor_actions)
            case "UCB":
                return self.getUCBAction()
            case _:
                return None