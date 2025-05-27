import numpy as np

class Environnement:
    def __init__(self, matrices, noise_dist, noise_params):
        self.agents = []
        self.matrices = matrices
        self.noise_dist = noise_dist
        self.noise_params = noise_params

    def ajouter_agents(self, agent):
        self.agents.append(agent)

    def sample_noise(self):
        if self.noise_dist == 'uniform':
            # TODO - est-ce que le noise_params (le 0, 0.1, 1) est utilisé ici ou plutôt dans les algos TS et KLUCB?
            low, high = self.noise_params
            return np.random.uniform(low, high)
        elif self.noise_dist == 'normal':
            mean, std = self.noise_params
            return np.random.normal(mean, std)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def updateStep(self, a1, a2):
        r1 = self.matrices[0][a1, a2] + self.sample_noise()
        r2 = self.matrices[1][a1, a2] + self.sample_noise()
        self.agents[0].update(a1, r1)
        self.agents[1].update(a2, r2)

    def step(self):
        action1, action2 = self.agents[0].train(), self.agents[1].train()
        self.updateStep(action1, action2)
        return [action1, action2]