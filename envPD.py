import numpy as np
from environment import Environnement

class EnvPD(Environnement):
    def __init__(self, matrices, agent_count, noise_dist='uniform', noise_params=(0.0, 0.05)):
        super().__init__(agent_count)
        if agent_count != len(matrices):
            raise Exception('Number of agents does not match number of matrices')
        
        self.matrices = matrices
        self.noise_dist = noise_dist
        self.noise_params = noise_params

    def sample_noise(self):
        if self.noise_dist == 'uniform':
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
        action1, action2 = self.agents[0].train([self.actions[1]]), self.agents[1].train([self.actions[0]])
        self.actions[0], self.actions[1] = action1, action2
        self.updateStep(action1, action2)
        return {'a1': action1, 'a2': action2}