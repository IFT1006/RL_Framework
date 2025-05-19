from execute import Execute
from utils import *

def main():
    # define the win rate - index 0 should have the biggest value
    win_rate = [0.5, 0.4]
    result = Execute(300, 1000, 11, win_rate, 2, "TUCB").getResult()
    avg = np.array(result['avg'])
    std = np.array(result['std'])
    printMean(avg, std)

    result = Execute(1, 1000, 11, win_rate, 2, "TUCB").getResult()
    printRuns(result['agents'])

if __name__ == '__main__':
    main()