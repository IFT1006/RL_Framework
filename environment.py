class Environment():
    def __init__(self, n_arms, n_neighbors, n_agents):
        if n_neighbors != n_agents - 1:
            print("Error: n_neighbors must equal n_agents - 1")

        # number of target plays (will be averaged over neighbours)
        self.target_plays = [0] * n_arms
        # number of own plays for arms 0 and 1
        self.plays = [0] * n_arms
        # average reward
        self.avg_reward = [0] * n_arms
        # step (play) number
        self.t = 0
        self.n_arms = n_arms
        self.n_neighbors = n_neighbors
