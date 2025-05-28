import numpy as np
from execute import Execute
from utils import plot_mean_std, plot_mean_std_2
import matplotlib.pyplot as plt
import seaborn as sns

# Définir les matrices de jeu
PDA = [[3, 5],
       [0, 1]]
PDB = [[3, 0],
       [5, 1]]

matrices = [np.array(PDA), np.array(PDB)]

# Paramètres d’expérience

noise_dist = "normal"
noise_params = [0.0, 0.1]



# #plt.rcParams.update({
#     "text.usetex":        False,
#     "font.family":        "serif",
#     "font.serif":         ["Times New Roman"],
#     "figure.dpi":         300,
#     "axes.titlesize":     4,
#     "axes.labelsize":     4,
#     "xtick.labelsize":    4,
#     "ytick.labelsize":    4,
#     "legend.fontsize":    4,
# })
sns.set_theme(style="whitegrid", palette="colorblind")


# Lancer une expérience
restot = {}

title = "PD_UCB2_TS0_N0_1000_test"
noise_exp = [[0.0, 0.0],[0.0, 0.1],[0.0, 1.0]]
algos = [["UCB", "UCB"],["TS", "TS"],["UCB", "TS"],["KLUCB", "KLUCB"]]
for placement_legende in ["lower right", "upper left"]:
    fig, axes = plt.subplots(
        1, len(noise_exp),
        sharey=True,
        figsize=(6.5, 2.5),  # largeur, hauteur en pouces
        dpi=300
    )
    for ax, noise in zip(axes,noise_exp):
        for algo_paired in algos:
            title = f"{algo_paired[0]}_vs_{algo_paired[1]}_{noise}"
            res = (Execute(10,1000,2,[2, 0], title).
                   getPDResult(matrices, algo_paired, 'normal', noise))
            restot.update({ title: res})

            mean = restot[title]['metrics']['mean_cum_regret']['agent_0']
            std = restot[title]['metrics']['std_cum_regret']['agent_0']
            x=np.arange(len(mean))
            ax.plot(
                x,
                mean,
                label=f"{algo_paired[0]}_vs_{algo_paired[1]}",
                linewidth=0.5,
                marker="o",
                markevery=30,
                markersize=2
            )
            ax.fill_between(
                x,
                mean,
                mean+ std,
                alpha=0.3
            )
        ax.set_title(f"$\\sigma_{{noise}}={noise}$")
        ax.set_xlabel("Time step (t)")
        # on ne remet pas ylabel à chaque fois car sharey=True
        ax.legend(loc=placement_legende, frameon=False)
        sns.despine(ax=ax, trim=True)

# 3) Labels communs
axes[0].set_ylabel("Mean cumulative regret $R(t)$")
plt.tight_layout()
fig.savefig("Workshop/Figure/side_by_side_noise.pdf")
plt.close(fig)
