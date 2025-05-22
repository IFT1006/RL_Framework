import numpy as np
from environment import Environnement

class EnvPD(Environnement):
    def __init__(self, matrices, agent_count, algo):
        super().__init__(agent_count, algo)
        if agent_count != len(matrices):
            raise Exception('Number of agents does not match number of matrices')
        self.matrice_jouer_A = matrices[0]
        self.matrice_jouer_B = matrices[1]

    def stepUCB(self):
        action1, action2 = self.agents[0].train(), self.agents[1].train()

        self.updateStep(action1, action2)
        return {'a1': action1, 'a2': action2}

    def stepTUCB(self):
        action1, action2 = self.agents[0].train([self.actions[1]]), self.agents[1].train([self.actions[0]])
        self.actions[0], self.actions[1] = action1, action2
        self.updateStep(action1, action2)
        return {'a1': action1, 'a2': action2}

    def updateStep(self, a1, a2):
        self.agents[0].update(a1, self.matrice_jouer_A[a1, a2] + np.random.normal(0., 0.3))
        self.agents[1].update(a2, self.matrice_jouer_B[a1, a2] + np.random.normal(0., 0.3))

    def step(self):
        if self.algo == 'UCB':
            return self.stepUCB()
        if self.algo == 'TUCB':
            return self.stepTUCB()
        return None