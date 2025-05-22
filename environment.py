class Environnement:
    def __init__(self, agent_count, algo):
        self.agents = []
        self.agent_count = agent_count
        self.algo = algo
        self.actions = [0] * agent_count

    def ajouter_agents(self, agent):
        self.agents.append(agent)
