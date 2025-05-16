import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
from learningAlgo import LearningAlgo

def main():
    # define the win rate - index 0 should have the biggest value
    win_rate = [0.5, 0.4]

    # initialize the list of agents and actions
    agents = []
    actions = []
    for i in range(0, 11):
        env = Environment(len(win_rate), 10, 11)
        learning_algo = LearningAlgo(2, "TUCB", env)
        agents.append(Agent(env, learning_algo))
        actions.append(0)

    # get 100 runs
    for t in range(0, 1000):
        # copy list of previous action
        prev_act = list(actions)

        for i in range(len(agents)):
            actions[i] = agents[i].train(win_rate, prev_act[0:i]+prev_act[(i+1):])

    plt.figure(figsize=(12, 8))
    i = 1
    for a in agents:
        plt.plot(a.cumul_regret, label="Agent " + str(i))
        i += 1
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Cumulative Regret of 4 TUCB Agents in a Fully Connected Graph", fontsize=20)
    plt.show()

if __name__ == '__main__':
    main()