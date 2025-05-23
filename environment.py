class Environnement:
    def __init__(self, agent_count):
        self.agents = []
        self.agent_count = agent_count
        self.actions = [0] * agent_count

    def ajouter_agents(self, agent):
        self.agents.append(agent)
