import numpy as np
from execute import Execute
from utils import plot_mean_std, plot_mean_std_2

# Définir les matrices de jeu
PDA = [[3, 5],
       [0, 1]]
PDB = [[3, 0],
       [5, 1]]

matrices = [np.array(PDA), np.array(PDB)]

# Paramètres d’expérience
horizon = 1000
algos = ["UCB", "TS"]
noise_dist = "normal"
noise_params = [0.0, 0.1]


# Lancer une expérience
restot = {}
title = "PD_UCB2_TS0_N0_1000_test"
res = Execute(
    n_instance = 500,
    T=horizon,
    n_agents=2,
    title=title
).getPDResult(
    matrices=matrices,
    algo=algos,
    noise_dist='normal',
    noise_params=tuple(noise_params)
)
restot.update({ title: res })

mean = restot[title]['metrics']['mean_cum_regret']['agent_0']
std = restot[title]['metrics']['std_cum_regret']['agent_0']
plot_mean_std_2(mean, std, title)