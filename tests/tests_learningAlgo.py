from environment import Environment
from learningAlgo import LearningAlgo

def tests_initialisation_d_un_algo():
    env = Environment(2,3, 4)
    learning_algo = LearningAlgo(2, 'TUCB', env)

    assert (
        learning_algo.constant == 2
    ), "Erreur: la constante n'est pas correctement initialisée"
    assert (
        learning_algo.est_opt == [0, 0]
    ), "Erreur: la liste est_opt n'est pas correctement initialisée"
    assert (
        learning_algo.target_opt == [0, 0]
    ), "Erreur: la liste target_opt n'est pas correctement initialisée"
    assert (
        learning_algo.action_val == [0, 0]
    ), "Erreur: la liste action_val n'est pas correctement initialisée"
    assert (
        learning_algo.algo_name == "TUCB"
    ), "Erreur: le nom de l'algo n'est pas correctement initialisé."
    assert isinstance(
        learning_algo.env, Environment
    ), "Erreur: Le env de l'agent doit être une instance de Environment."

def tests_getTUCBAction():
    env = Environment(2,3, 4)
    learning_algo = LearningAlgo(2, 'TUCB', env)
    agent = Agent(env, learning_algo)
    for i in range(100):
        agent.train([0.6, 0.4], [0, 0, 0])
    assert (
        agent.env.t == 100
    ), "Erreur: Le nombre de fois jouées doit être 100 après 100 runs."

def tests():
    tests_initialisation_d_un_algo()
    tests_getTUCBAction()
