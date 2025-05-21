from execute import Execute
from utils import *

def main():
    # define the fixed win rate - index 0 should have the biggest value
    win_rate = [0.5, 0.4]

    result = Execute(300, 1000, 4, win_rate, 2, "TUCB", False).getBanditResult()
    avg = np.array(result['avg'])
    std = np.array(result['std'])
    printMean(avg, std, 4, 300, 'TUCB')
    printRuns(result['agents'])

    result = Execute(500, 1000, 11, win_rate, 2, "TUCB", False).getBanditResult()
    avg = np.array(result['avg'])
    std = np.array(result['std'])
    printMean(avg, std, 11, 500, "TUCB")

    result = Execute(50, 5000, 1, win_rate, 2, "UCB", True).getBanditResult()
    avg = np.array(result['avg'])
    std = np.array(result['std'])
    printMean(avg, std, 1, 50, "UCB")

    # result = Execute(50, 5000, 2, win_rate, 2, "UCB", False).getPDResult()
    # avg = np.array(result['avg'])
    # std = np.array(result['std'])
    # printMean(avg, std, 2, 50, "UCB")

if __name__ == '__main__':
    main()