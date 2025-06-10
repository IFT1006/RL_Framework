from execute import Execute
from utils import *
import numpy as np

def main():
    # define the fixed win rate - index 0 should have the biggest value
    win_rate = [0.5, 0.4]

    result = Execute(300, 1000, 4, 1.5).getBanditResult(win_rate, False, "TUCB")
    avg = np.array(result['avg'])
    std = np.array(result['std'])
    printMean(avg, std, 4, 300, 'TUCB')
    printRuns(result['agents'], "Cumulative Regret of 4 TUCB Agents in a Fully Connected Graph")

    result = Execute(500, 1000, 11, 1.5).getBanditResult(win_rate, False, "TUCB")
    avg = np.array(result['avg'])
    std = np.array(result['std'])
    printMean(avg, std, 11, 500, "TUCB")

    result = Execute(50, 5000, 1, 2).getBanditResult(win_rate, True, "UCB")
    avg = np.array(result['avg'])
    std = np.array(result['std'])
    printMean(avg, std, 1, 50, "UCB")

    # A_PD = np.array([[0.5, 0],
    #                  [1, 0.2]]).astype(float)
    # B_PD = np.array([[0.5, 1],
    #                  [0, 0.2]]).astype(float)
    # matrices = [A_PD, B_PD]
    # result = Execute(5000, 1000, 2, 2).getPDResult(matrices, ["UCB", "UCB"])
    # prop = np.array(result['prop'])
    # printProp(prop, "Prop of action2 for agent2 in Prisoner's Dilemma UCB [0.5,0.2] w/ init & noise std .3")
    #
    # result = Execute(5000, 1000, 2, 1.5).getPDResult(matrices, ["TUCB", "TUCB"])
    # prop = np.array(result['prop'])
    # printProp(prop, "Prop of action2 for agent2 in Prisoner's Dilemma TUCB [0.5,0.2] w/ init & noise std .3")

    # A_PD = np.array([[1, 0, 0.5],
    #                  [0, 1, 0.5],
    #                  [0.5, 0.5, 0.5]]).astype(float)
    # B_PD = A_PD
    # matrices = [A_PD, B_PD]
    #
    # result = Execute(5000, 1000, 2, 2).getPDResult(matrices, ["UCB", "UCB"])
    # props = np.array(result['prop'])
    # printProp3(props, "Prop of each action for agent2 in 3-dim matrix UCB w/ init & noise uni .05")
    #
    # result = Execute(5000, 1000, 2, 1.5).getPDResult(matrices, ["TUCB", "TUCB"])
    # props = np.array(result['prop'])
    # printProp3(props, "Prop of each action for agent2 in 3-dim matrix TUCB w/ init & noise uni .05")

    ############
    # k = -3
    # A_PG = np.array([[10, 0, k],
    #                  [0, 2, 0],
    #                  [k, 0, 10]]).astype(float)
    # B_PG = A_PG
    # A_CG = np.array([[11, -30, 0],
    #                  [-30, 7, 6],
    #                  [0, 0, 5]]).astype(float)
    # B_CG = A_CG

    # etendu_unif = 0.05
    # A = normalizeMatrix(A_PG, etendu_unif)
    # B = normalizeMatrix(B_PG, etendu_unif)
    # matrices = [A, B]

    # result = Execute(500, 200, 2, 2).getPDResult(matrices, ["UCB", "UCB"])
    # printProp3(np.array(result['prop']), "Prop of each action for agent2 in 3-dim PG UCB vs UCB w/ init noise: U(0,0.1)")

    # result = Execute(500, 200, 2, 1.5).getPDResult(matrices, ["TUCB", "TUCB"])
    # props = np.array(result['prop'])
    # printProp3(props, "Prop of each action for agent2 in 3-dim PG TUCB vs TUCB w/ init noise: U(0,0.1)")

    # result = Execute(500, 200, 2, 2).getPDResult(matrices, ["UCB", "TUCB"])
    # props = np.array(result['prop'])
    # printProp3(props, "Prop of each action for agent2 in 3-dim PG TUCB vs UCB w/ init noise: U(0,0.1)")

    # A = normalizeMatrix(A_CG, etendu_unif)
    # B = normalizeMatrix(B_CG, etendu_unif)
    # matrices = [A, B]

    # result = Execute(500, 200, 2, 2).getPDResult(matrices, ["UCB", "UCB"])
    # printProp3(np.array(result['prop']), "Prop of each action for agent2 in 3-dim CG UCB vs UCB w/ init noise: U(0,0.1)")

    # result = Execute(500, 200, 2, 1.5).getPDResult(matrices, ["TUCB", "TUCB"])
    # props = np.array(result['prop'])
    # printProp3(props, "Prop of each action for agent2 in 3-dim CG TUCB vs TUCB w/ init noise: U(0,0.1)")

    # result = Execute(500, 200, 2, 2).getPDResult(matrices, ["UCB", "TUCB"])
    # props = np.array(result['prop'])
    # printProp3(props, "Prop of each action for agent2 in 3-dim CG TUCB vs UCB w/ init noise: U(0,0.1)")
    ######

if __name__ == '__main__':
    main()