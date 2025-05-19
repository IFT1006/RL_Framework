from agent import Agent
from environment import Environment
from learningAlgo import LearningAlgo

def tests_initialisation_d_un_agent():
    env = Environment(2,3, 4)
    learning_algo = LearningAlgo(2, 'TUCB', env)
    agent = Agent(env, learning_algo)

    assert (
        agent.cumul_regret == []
    ), "Erreur: la liste de regret n'est pas correctement initialisée"
    assert isinstance(
        agent.env, Environment
    ), "Erreur: Le env de l'agent doit être une instance de Environment."
    assert isinstance(
        agent.learning_algo, LearningAlgo
    ), "Erreur: Le learning_algo de l'agent doit être une instance de LearningAlgo."

def tests_train():
    env = Environment(2,3, 4)
    learning_algo = LearningAlgo(2, 'TUCB', env)
    agent = Agent(env, learning_algo)
    for i in range(100):
        agent.train([0.6, 0.4], [0, 0, 0])
    assert (
        agent.env.t == 100
    ), "Erreur: Le nombre de fois jouées doit être 100 après 100 runs."

def tests_winrate_premier_index():
    env = Environment(2, 3, 4)
    learning_algo = LearningAlgo(2, 'TUCB', env)
    agent = Agent(env, learning_algo)
    try:
        agent.train([0.4, 0.6], [0, 0, 0])
        assert False, "Error: index 0 must have the biggest value"
    except Exception:
        pass

def tests_update():
    env = Environment(2,3, 4)
    learning_algo = LearningAlgo(2, 'TUCB', env)
    agent = Agent(env, learning_algo)
    for i in range(50):
        agent.update(1, 0, [0.6, 0.4])
        agent.update(0, 1, [0.6, 0.4])
    assert (
        len(agent.cumul_regret) == 100
    ), "Erreur: La longueur de cumul_regret n'est pas correcte."
    assert(
        agent.env.plays == [50, 50]
    ), "Erreur: La liste de plays n'est pas correcte."
    assert (
        agent.env.avg_reward == [1, 0]
    ), "Erreur: La liste de avg_reward n'est pas correcte."

def tests():
    tests_initialisation_d_un_agent()
    tests_train()
    tests_winrate_premier_index()
    tests_update()
