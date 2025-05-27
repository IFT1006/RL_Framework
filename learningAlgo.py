import numpy as np
from agentSpace import AgentSpace

class LearningAlgo:
    def __init__(self, constant, algo_name, a_space: AgentSpace, noise_param):
        self.constant = constant
        self.algo_name = algo_name
        self.a_space = a_space
        self.init_sequence = np.random.permutation(a_space.n_arms)
        self.noise_param = noise_param

    def getInitialState(self):
        first_time = False
        action = 0
        # générer l'action aléatoirement pour l'initialisation
        if self.a_space.t <= self.a_space.n_arms:
            action = self.init_sequence[self.a_space.t-1]
            first_time = True

        return {'action': action, 'first_time': first_time }

    def getUCBAction(self, first_time, action):
        if not first_time:
            # start the algo after initialization
            est_opt = np.sqrt(self.constant * np.log(self.a_space.t) / self.a_space.plays)
            action_val = self.a_space.avg_reward + est_opt

            best = np.flatnonzero(action_val == action_val.max())
            if best.size == 1:
                action = int(best[0])
            else:
                action = int(np.random.choice(best))
        return action
    
    def getTSAction(self, first_time, action):
        # Mettre self. quand on va diviser les algos
        mu_0 = 1 # Lorsqu'on va diviser, il faut qu'on puisse modifier ça
        var_0 = 1 # Lorsqu'on va diviser, il faut qu'on puisse modifier ça
        var = max(self.noise_param, 1e-2)

        if not first_time:
            mu_post = (mu_0/var_0 + self.a_space.sums/var) / (1/var_0 + self.a_space.plays/var)
            var_post = 1 / (1 / var_0 + self.a_space.plays / var)
            samples = np.random.normal(mu_post, np.sqrt(var_post))

            best = np.flatnonzero(samples == samples.max())
            if best.size == 1:
                action = int(best[0])
            else:
                action = int(np.random.choice(best))
        return action

    def getKLUCBAction(self, first_time, action):
        var = max(self.noise_param, 1e-2)
        c = 3

        if not first_time:
            means = self.a_space.sums / self.a_space.plays
            f_t = 2 * var * (np.log(self.a_space.t) + c * np.log(np.log(self.a_space.t)))
            ucbs = means + np.sqrt(f_t / self.a_space.plays)

            best = np.flatnonzero(ucbs == ucbs.max())
            if best.size == 1:
                action = int(best[0])
            else:
                action = int(np.random.choice(best))
        return action

    def getAction(self):
        res = self.getInitialState()
        first_time = res['first_time']
        action = res['action']

        match self.algo_name:
            case "UCB":
                return self.getUCBAction(first_time, action)
            case "TS":
                return self.getTSAction(first_time, action)
            case "KLUCB":
                return self.getKLUCBAction(first_time, action)
            case _:
                return None