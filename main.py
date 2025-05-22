from execute import Execute
from utils import *
import numpy as np

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

    # A_PD = np.array([[0.5, 0],
    #                  [1, 0.2]]).astype(float)
    # B_PD = np.array([[0.5, 1],
    #                  [0, 0.2]]).astype(float)
    # matrices = [A_PD, B_PD]
    # result = Execute(5000, 1000, 2, 2, "UCB").getPDResult(matrices)
    # prop = np.array(result['prop'])
    # printProp(prop, "Prop of action2 for agent2 in Prisoner's Dilemma UCB [0.5,0.2] w/ init & noise std .3")
    #
    # result = Execute(5000, 1000, 2, 2, "TUCB").getPDResult(matrices)
    # prop = np.array(result['prop'])
    # printProp(prop, "Prop of action2 for agent2 in Prisoner's Dilemma TUCB [0.5,0.2] w/ init & noise std .3")

    A_PD = np.array([[1, 0, 0.5],
                     [0, 1, 0.5],
                     [0.5, 0.5, 0.5]]).astype(float)
    B_PD = np.array([[1, 0, 0.5],
                     [0, 1, 0.5],
                     [0.5, 0.5, 0.5]]).astype(float)
    matrices = [A_PD, B_PD]

    result = Execute(5000, 1000, 2, 2, "UCB").getPDResult(matrices)
    props = np.array(result['prop'])
    printProp3(props, "Prop of each action for agent2 in 3-dim matrix UCB w/ init & noise std .3")

    result = Execute(5000, 1000, 2, 2, "TUCB").getPDResult(matrices)
    props = np.array(result['prop'])
    printProp3(props, "Prop of each action for agent2 in 3-dim matrix TUCB w/ init & noise std .3")


if __name__ == '__main__':
    main()