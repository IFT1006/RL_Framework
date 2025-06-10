from unittest import mock
from reward import Reward

def tests_initialisation_d_un_reward():
    reward = Reward([0.6, 0.4], 1)

    assert (
        reward.win_rate == [0.6, 0.4]
    ), "Erreur: la liste de win_rate n'est pas correctement initialisée"
    assert (
        reward.arm_played == 1
    ), "Erreur: Le arm_played n'est pas correctement initialisé."

@mock.patch('reward.np.random.rand')
def tests_get_reward(mocked):
    mocked.return_value = 0.5
    reward1 = Reward([0.6, 0.4], 1).getReward()
    assert (
        reward1 == 0
    ), "Erreur: La récompense doit être 0 pour action 1 lorsque rand et 0.5."
    reward0 = Reward([0.6, 0.4], 0).getReward()
    assert (
        reward0 == 1
    ), "Erreur: La récompense doit être 1 pour action 0 lorsque rand et 0.5."


def tests():
    tests_initialisation_d_un_reward()
    tests_get_reward()