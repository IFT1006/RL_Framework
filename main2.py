import numpy as np
from execute import Execute
from utils import plot_mean_std, plot_mean_std_2
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(43)

plt.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.serif":         ["Times New Roman"],
    "figure.dpi":         300,
    "axes.titlesize":     4,
    "axes.labelsize":     4,
    "xtick.labelsize":    4,
    "ytick.labelsize":    4,
    "legend.fontsize":    4,
})
sns.set_theme(style="whitegrid", palette="colorblind")


# Lancer une expérience
# restot = {}
# SG = [[2,0],[0,1]]
# matrices = [np.array(SG), np.array(SG)]
# title = "PG_3_UCB2_TS0_N0_1000_test"
# noise_exp = [[0.0, 0.0],[0.0, 0.1],[0.0, 1.0]]
# algos = [["UCB", "UCB"],["TS", "TS"],["UCB", "TS"],["KLUCB", "KLUCB"]]
# #for placement_legende in ["lower right", "upper left"]:
# sharey = False
# fig, axes = plt.subplots(1,3,sharey=sharey, figsize=(15, 4))
# for ax, noise in zip(axes,noise_exp):
#     for algo_paired in algos:
#         title = f"{algo_paired[0]}_vs_{algo_paired[1]}_{noise}"
#         res = (Execute(500,1000,2,[None, None], title).
#                getPDResult(matrices, algo_paired, 'normal', noise))
#         restot.update({ title: res})
#
#         mean = restot[title]['metrics']['mean_cum_regret']['agent_0']
#         std = restot[title]['metrics']['std_cum_regret']['agent_0']
#         x=np.arange(len(mean))
#         ax.plot(
#             x,
#             mean,
#             label=f"{algo_paired[0]}_vs_{algo_paired[1]}",
#             linewidth=1,
#         )
#         ax.fill_between(
#             x,
#             mean,
#             mean+ std,
#             alpha=0.2
#         )
#
#     ax.set_title(f"$\\sigma_{{noise}}={noise[1]}$")
#     ax.set_xlabel("Time step (t)")
#     sns.despine(ax=ax, trim=True)
#
# # 3) Labels communs
#
# # Version 1
# axes[0].set_ylim(axes[1].get_ylim())
# axes[1].set_ylim(axes[1].get_ylim())
# axes[2].set_ylim(axes[2].get_ylim())
#
# # # Version 2
# # axes[0].set_ylim(axes[0].get_ylim())
# # axes[1].set_ylim(axes[1].get_ylim())
# # axes[2].set_ylim(axes[2].get_ylim())
#
# sns.set_style("whitegrid")
# for ax in axes:
#     ax.grid(alpha=0.3)
#
# handles, labels = axes[0].get_legend_handles_labels()
# axes[0].legend(handles, labels, loc="upper left", frameon=False)
#
# # only the leftmost gets the y‑axis label
# axes[0].set_ylabel("Mean cumulative regret $R(t)$")
# axes[0].set_ylabel("Mean cumulative regret $R(t)$")
# plt.tight_layout()
# fig.savefig("Workshop/Figure/SG_v1.pdf", dpi=300, bbox_inches="tight")
# plt.close(fig)


#######################################################################################################################
PD
#######################################################################################################################
PD = [[0.6,0],[0,1]]
matrices = [np.array(PD), np.array(PD).T]

restot = {}

title = "PD"
noise_exp = [[0.0, 0.0],[0.0, 0.1],[0.0, 1.0]]
algos = [["UCB", "UCB"],["TS", "TS"],["UCB", "TS"],["KLUCB", "KLUCB"]]
#for placement_legende in ["lower right", "upper left"]:
sharey = False
fig, axes = plt.subplots(1,3,sharey=sharey, figsize=(15, 4))
for ax, noise in zip(axes,noise_exp):
    for algo_paired in algos:
        title = f"{algo_paired[0]}_vs_{algo_paired[1]}_{noise}"
        res = (Execute(500,1000,2,[None, None], title).
               getPDResult(matrices, algo_paired, 'normal', noise))
        restot.update({ title: res})

        mean = restot[title]['metrics']['mean_cum_regret']['agent_0']
        std = restot[title]['metrics']['std_cum_regret']['agent_0']
        x=np.arange(len(mean))
        ax.plot(
            x,
            mean,
            label=f"{algo_paired[0]}_vs_{algo_paired[1]}",
            linewidth=1,
        )
        ax.fill_between(
            x,
            mean,
            mean+ std,
            alpha=0.2
        )

    ax.set_title(f"$\\sigma_{{noise}}={noise[1]}$")
    ax.set_xlabel("Time step (t)")
    sns.despine(ax=ax, trim=True)

# 3) Labels communs

# Version 2
axes[0].set_ylim(axes[0].get_ylim())
axes[1].set_ylim(axes[1].get_ylim())
axes[2].set_ylim(axes[2].get_ylim())

sns.set_style("whitegrid")
for ax in axes:
    ax.grid(alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc="upper left", frameon=False)

# only the leftmost gets the y‑axis label
axes[0].set_ylabel("Mean cumulative regret $R(t)$")
axes[0].set_ylabel("Mean cumulative regret $R(t)$")
plt.tight_layout()
fig.savefig("Workshop/Figure/PD_v1.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)















