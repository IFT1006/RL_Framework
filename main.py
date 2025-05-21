from execute import Execute
from utils import *

def main():
    # define the fixed win rate - index 0 should have the biggest value
    # win_rate = [0.5, 0.4]
    #
    # result = Execute(300, 1000, 4, 2, "TUCB").getBanditResult(win_rate, False)
    # avg = np.array(result['avg'])
    # std = np.array(result['std'])
    # printMean(avg, std, 4, 300, 'TUCB')
    # printRuns(result['agents'], "Cumulative Regret of 4 TUCB Agents in a Fully Connected Graph")
    #
    # result = Execute(500, 1000, 11, 2, "TUCB").getBanditResult(win_rate, False)
    # avg = np.array(result['avg'])
    # std = np.array(result['std'])
    # printMean(avg, std, 11, 500, "TUCB")
    #
    # result = Execute(50, 5000, 1, 2, "UCB").getBanditResult(win_rate, True)
    # avg = np.array(result['avg'])
    # std = np.array(result['std'])
    # printMean(avg, std, 1, 50, "UCB")

    result = Execute(5000, 1000, 2, 2, "UCB").getPDResult()
    prop = np.array(result['prop'])
    printProps(prop, "Proportion of action 2 played for agent 2 in a Prisoner's Dilemma game UCB")

    result = Execute(5000, 1000, 2, 2, "TUCB").getPDResult()
    prop = np.array(result['prop'])
    printProps(prop, "Proportion of action 2 played for agent 12 in a Prisoner's Dilemma game TUCB")

if __name__ == '__main__':
    main()