import numpy as np
class AgentSpace:
    def __init__(self, n_arms, n_agents, game, a_id):
        # number of target plays (will be averaged over neighbours)
        self.target_plays = np.zeros(n_arms, dtype=int)
        # number of own plays for arms 0 and 1
        self.plays = np.zeros(n_arms, dtype=int)
        # average reward
        self.avg_reward = np.zeros(n_arms, dtype=int)
        # step (play) number
        self.t = 0
        self.n_arms = n_arms
        self.n_neighbors = n_agents - 1
        self.game = game
        self.a_id = a_id
