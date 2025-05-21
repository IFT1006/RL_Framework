import numpy as np, random

class Environnement:
    def __init__(self, matrices, agent_count):
        if agent_count != len(matrices):
            raise Exception('Number of agents does not match number of matrices')
        self.matrices = matrices
        self.agents = []
        self.agents_count = agent_count

    def ajouter_agents(self, agent):
        self.agents.append(agent)

    def step(self):
        for i in range(0, len(self.agents)-1, 2):
            ag1, ag2 = self.agents[i], self.agents[i+1]
            a1, a2 = ag1.play(), ag2.play()      # entiers 0/1

            ag1.update(self.A[a1,a2])
            ag2.update(self.B[a1,a2])