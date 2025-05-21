from environment import Environnement

class EnvPD(Environnement):
    def __init__(self, matrices, agent_count):
        super().__init__(agent_count)
        if agent_count != len(matrices):
            raise Exception('Number of agents does not match number of matrices')
        self.matrice_jouer_A = matrices[0]
        self.matrice_jouer_B = matrices[1]

    def step(self):
        action1, action2 = self.agents[0].train(), self.agents[1].train()

        self.agents[0].update(action1, self.matrice_jouer_A[action1, action2])
        self.agents[1].update(action2, self.matrice_jouer_B[action1, action2])