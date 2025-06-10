from agentSpace import AgentSpace

def tests_initialisation_d_un_environnement():
    env = AgentSpace(2, 3, 4)

    assert (
        env.plays == [0, 0]
    ), "Erreur: la liste de plays n'est pas correctement initialisée"
    assert (
        env.target_plays == [0, 0]
    ), "Erreur: la liste de target_plays n'est pas correctement initialisée."
    assert (
        env.avg_reward == [0, 0]
    ), "Erreur: la liste de avg_reward n'est pas correctement initialisée."
    assert (
        env.t == 0
    ), "Erreur: le nombre de fois jouées n'est pas correctement initialisée."
    assert (
        env.n_arms == 2
    ), "Erreur: le nombre de bras n'est pas correctement initialisée."
    assert (
        env.n_neighbors == 3
    ), "Erreur: le nombre de voisins n'est pas correctement initialisée."

def tests_nombre_de_voisins():
    try:
        AgentSpace(2, 4, 4)
        assert False, "Error: n_neighbors must equal n_agents - 1"
    except Exception:
        pass

def tests():
    tests_initialisation_d_un_environnement()
    tests_nombre_de_voisins()

