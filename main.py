import matplotlib.pyplot as plt
import numpy as np
from execute import Execute

def main():
    # define the win rate - index 0 should have the biggest value
    win_rate = [0.5, 0.4]
    result = Execute(2000, 1000, 11, win_rate, 2, "TUCB").getResult()

    avg = np.array(result['avg'])
    std = np.array(result['std'])

    plt.figure(figsize=(12, 8))
    # i = 1
    # for a in agents:
    x = np.arange(len(avg))
    plt.plot(x, avg, label="Agent " + str(1))
    plt.fill_between(x, avg, avg + std, alpha=0.2)
    #     i += 1
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Cumulative Regret of 4 TUCB Agents in a Fully Connected Graph", fontsize=20)
    plt.show()

if __name__ == '__main__':
    main()