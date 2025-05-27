import numpy as np
class AgentSpace:
    def __init__(self, n_arms, n_agents, game, a_id):
        # number of target plays (will be averaged over neighbours)
        self.target_plays = np.zeros(n_arms, dtype=int)
        self.plays = np.zeros(n_arms, dtype=int)
        self.avg_reward = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        # step (play) number
        self.t = 0
        self.n_arms = n_arms
        self.n_neighbors = n_agents - 1
        self.game = game
        self.a_id = a_id
        self.weight = np.ones(n_arms)
        self.hist_probas = np.zeros(n_arms)
