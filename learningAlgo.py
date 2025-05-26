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
        self.init_sequence = np.random.permutation(self.a_space.n_arms)

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
            # générer l'action aléatoirement pour l'initialisation
            if self.a_space.t <= self.a_space.n_arms:
                action = self.init_sequence[self.a_space.t]

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

            best = np.flatnonzero(action_val == action_val.max())
            if best.size == 1:
                action = int(best[0])
            else:
                #print("Tie_TUCB")
                #print('action_val', action_val)
                action = int(np.random.choice(best))
        return action

    def getUCBAction(self, first_time, action):
        if not first_time:

            # start the algo after initialization
            action_val = np.zeros(self.a_space.n_arms)
            est_opt = np.sqrt(self.constant * np.log(self.a_space.t) / self.a_space.plays)
            action_val = self.a_space.avg_reward + est_opt

            best = np.flatnonzero(action_val == action_val.max())
            if best.size == 1:
                action = int(best[0])
            else:
                #print("Tie_UCB")
                #print('action_val', action_val)
                action = int(np.random.choice(best))
        return action
    
    def getTSAction(self, first_time, action):

        # Mettre self. quand on va diviser les algos
        mu_0 = 1 # Lorsqu'on va diviser, il faut qu'on puisse modifier ça
        var_0 = 1 # Lorsqu'on va diviser, il faut qu'on puisse modifier ça
        # on utilise self.constant ici pour le noise_var dans l'algo. À changer pt
        var = max(self.constant, 1e-2)

        if not first_time:
            mu_post = (mu_0/var_0 + self.a_space.sums/var) / (1/var_0 + self.a_space.plays/var)
            var_post = 1 / (1 / var_0 + self.a_space.plays / var)
            samples = np.random.normal(mu_post, np.sqrt(var_post))

            best = np.flatnonzero(samples == samples.max())
            if best.size == 1:
                action = int(best[0])
            else:
                #print("Tie_TS")
                #print('samples', samples)
                action = int(np.random.choice(best))
        return action

    def getKLUCBAction(self, first_time, action):
        # on utilise self.constant ici pour le noise_var dans l'algo. À changer pt
        var = max(self.constant, 1e-2)
        c = 3

        if not first_time:
            means = self.a_space.sums / self.a_space.plays
            f_t = 2 * var * (np.log(self.a_space.t) + c * np.log(np.log(self.a_space.t)))
            ucbs = means + np.sqrt(f_t / self.a_space.plays)

            best = np.flatnonzero(ucbs == ucbs.max())
            if best.size == 1:
                action = int(best[0])
            else:
                #print("Tie_TS")
                #print('samples', samples)
                action = int(np.random.choice(best))
        return action

    def getEpsilonGreedyAction(self, first_time, action):
        if not first_time:
            # on utilise self.constant pour représenter l'epsilon dans l'algo. À changer pt
            if self.constant is None:
                epsilon = 1 / np.sqrt(self.a_space.t)
            else:
                epsilon = self.constant

            if np.random.rand() < epsilon:
                action = np.random.choice(self.a_space.n_arms)
            else:
                means = self.a_space.sums / self.a_space.plays
                best = np.flatnonzero(means == means.max())
                if best.size == 1:
                    action = int(best[0])
                else:
                    # print("Tie_TS")
                    # print('samples', samples)
                    action = int(np.random.choice(best))

        return action

    def getExp3Action(self, first_time, action):
        if not first_time:
            total_w = sum(self.a_space.weight)

            actions = list(range(self.a_space.n_arms))
            pA = (1 - self.constant) * (self.a_space.weight / total_w) + self.constant / self.a_space.n_arms
            self.a_space.hist_probas.append(pA)
            action = random.choices(actions, weights=pA, k=1)[0]

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
            case "TS":
                return self.getTSAction(first_time, action)
            case "KLUCB":
                return self.getKLUCBAction(first_time, action)
            case "EpsilonGreedy":
                return self.getEpsilonGreedyAction(first_time, action)
            case "Exp3":
                return self.getExp3Action(first_time, action)
            case _:
                return None