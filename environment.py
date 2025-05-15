class Environment():
    def __init__(self, nbr_actions, nbr_neighbors):
        # number of target plays (averaged over neighbours)
        self.target_plays = [0] * nbr_actions
        self.nbr_actions = nbr_actions
        # number of own plays for arms 0 and 1
        self.plays = [0]*nbr_actions
        self.nbr_neighbors = nbr_neighbors
