import matplotlib.pyplot as plt
import numpy as np


def normalizeMatrix(matrix, etendue):
    # Normalisation for [0,1]
    matrix_norm = (matrix-np.min(matrix))/np.ptp(matrix)
 
    # Normalisation considering the noise
    matrix_norm_noise = matrix_norm*(1-etendue)+etendue/2
    return matrix_norm_noise

def plot_mean_std(df,title, agent=0, label=None):

    time  = df['step']
    mean  = df[f'mean_cum_regret_agent_{agent}']
    std   = df[f'std_cum_regret_agent_{agent}']
    lbl   = label or f'Agent {agent}'

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, mean, label=lbl)
    ax.fill_between(time, mean, mean + std, alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean cumul regret')
    ax.legend()

    fig.savefig(f'Workshop/Figure/{title}_agent_0.pdf',format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_mean_std_2(mean, std, title, agent=0, label=None):

    time  = [i+1 for i in range(len(mean))]
    lbl   = label or f'Agent {agent}'

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, mean, label=lbl)
    ax.fill_between(time, mean, mean + std, alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean cumul regret')
    ax.legend()

    fig.savefig(f'Workshop/Figure/{title}_agent_0.pdf',format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)






def printGraph(str, ylabel):
    plt.xlabel("Plays", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
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

    printGraph("Mean of cumulative Regret of "+algo+" Agent1 in a Fully Connected Graph", "Cumulative regret")

def printRuns(agents, show_str):
    plt.figure(figsize=(12, 8))
    i = 1
    for a in agents:
        plt.plot(a.cumul_regret, label="Agent " + str(i))
        i += 1
    printGraph(show_str, "Cumulative regret")

def printProp(prop, show_str):
    plt.figure(figsize=(12, 8))
    plt.plot(prop, label="Agent 2")
    printGraph(show_str, "Proportion of action 2")

def printProp3(action, show_str):
    plt.figure(figsize=(12, 8))
    i = 1
    for a in action:
        plt.plot(a, label="Action " + str(i))
        i += 1
    printGraph(show_str, "Proportion of each action")



