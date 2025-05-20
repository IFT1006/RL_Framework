import matplotlib.pyplot as plt
import numpy as np

def printGraph(str):
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel("Cumulative regret", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title(str, fontsize=20)
    plt.show()

def printMean(avg, std, n_agents, n_ins, algo):
    plt.figure(figsize=(12, 8))

    x = np.arange(len(avg))
    plt.plot(x, avg, label="Agent " + str(1))
    plt.fill_between(x, avg, avg + std, alpha=0.2)
    plt.text(.01, .99, str(n_agents)+' agents;'+' '+str(n_ins)+' instances')

    printGraph("Mean of cumulative Regret of "+algo+" Agent 1 in a Fully Connected Graph")

def printRuns(agents):
    plt.figure(figsize=(12, 8))
    i = 1
    for a in agents:
        plt.plot(a.cumul_regret, label="Agent " + str(i))
        i += 1

    printGraph("Cumulative Regret of 4 TUCB Agents in a Fully Connected Graph")