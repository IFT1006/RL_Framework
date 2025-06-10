from unittest.mock import MagicMock, patch

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo

def tests_initialisation_d_un_algo():
    env = AgentSpace(2, 3, 4)
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
        learning_algo.env, AgentSpace
    ), "Erreur: Le env de l'agent doit être une instance de Environment."

environment = AgentSpace(2, 3, 4)
@patch.object(environment, 't', 2)
def tests_getTUCBActionT():
    learning_algo = LearningAlgo(2, 'TUCB', environment)

    learning_algo.getTUCBAction([0, 0, 0])

    assert (
        learning_algo.env.target_plays[0] == 1
    ), "Erreur: les valeurs dans Target_plays doivent commencer à incrémenter à partir de la 2e itération."

@patch.object(environment, 'plays', [0, 0])
def tests_getTUCBActionPlay0():
    learning_algo = LearningAlgo(2, 'TUCB', environment)

    action = learning_algo.getTUCBAction([0, 0, 0])

    assert (
            action == 0
    ), "Erreur: l'action pour la 1e itération doit être 0'."

@patch.object(environment, 'plays', [1, 0])
def tests_getTUCBActionPlay1():
    learning_algo = LearningAlgo(2, 'TUCB', environment)

    action = learning_algo.getTUCBAction([0, 0, 0])

    assert (
            action == 1
    ), "Erreur: l'action pour la 2e itération doit être 1'."

def tests_getAction():
    env = AgentSpace(2, 3, 4)
    learning_algo = LearningAlgo(2, 'TUCB', env)

    learning_algo.getTUCBAction = MagicMock()
    learning_algo.getAction([0, 0, 0])
    learning_algo.getTUCBAction.assert_called_once()

def tests_getActionNotCalled():
    env = AgentSpace(2, 3, 4)
    learning_algo = LearningAlgo(2, 'UCB', env)

    learning_algo.getTUCBAction = MagicMock()
    learning_algo.getAction([0, 0, 0])
    learning_algo.getTUCBAction.assert_not_called()

def tests():
    tests_initialisation_d_un_algo()
    tests_getTUCBActionT()
    tests_getTUCBActionPlay0()
    tests_getTUCBActionPlay1()
    tests_getAction()
    tests_getActionNotCalled()
