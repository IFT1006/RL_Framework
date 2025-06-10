import tests_agent
import tests_environment
import tests_reward
import tests_learningAlgo

if __name__ == "__main__":
    print("Tests unitaires de l'ensemble du projet...")

    tests_agent.tests()
    tests_environment.tests()
    tests_reward.tests()
    tests_learningAlgo.tests()

    print("Tests unitaires passés avec succès!")
