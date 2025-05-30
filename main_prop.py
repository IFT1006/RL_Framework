import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from execute import Execute
from tqdm import tqdm
np.random.seed(43)

plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.serif":      ["Times New Roman"],
    "figure.dpi":      300,
    "axes.titlesize":  5,
    "axes.labelsize":  5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
})
sns.set_theme(style="whitegrid", palette="colorblind")

# 1) Définition des matrices de chaque jeu
games = {
    "PG_wp": [np.array([[1,0.1,0],[0.1,0.28,0.1],[0,0.1,1]]),
              np.array([[1,0.1,0],[0.1,0.28,0.1],[0,0.1,1]])],
    "PG":    [np.array([[1,0,0],[0,0.2,0],[0,0,1]]),
              np.array([[1,0,0],[0,0.2,0],[0,0,1]])],
    "PD":    [np.array([[0.6,0],[1,0.4]]),
              np.array([[0.6,0],[1,0.4]]).T],
    "SG":    [np.array([[1,0],[0,0.5]]),
              np.array([[1,0],[0,0.5]])],
    "CG_no": [np.array([[1,0,0.75],[0,0.9,0.85],[0.75,0.75,0.85]]),
              np.array([[1,0,0.75],[0,0.9,0.85],[0.75,0.75,0.85]])]
}

noise_lvls = [
    [0.0, 0.0],
    [0.0, 0.1],
    [0.0, 1.0]
]

algo_pairs = [
    ["UCB",   "UCB"],
    ["KLUCB", "KLUCB"],
    ["TS",    "TS"],
    ["UCB",   "KLUCB"],
    ["UCB",   "TS"]
]

n_runs = 500

# 2) Boucle sur jeux puis sur niveaux de bruit
for game_name, matrices in tqdm(games.items()):
    for noise in noise_lvls:
        # Création d’une nouvelle figure pour ce (jeu, bruit)
        fig, axes = plt.subplots(
            len(algo_pairs), 1,
            sharex=True, sharey=True,
            figsize=(6, 8),
            constrained_layout=True,
            dpi=300
        )

        for ax, (a1, a2) in zip(axes, algo_pairs):
            title = f"{game_name}_{a1}_vs_{a2}_noise{noise[1]:.2f}"
            res = Execute(n_runs, 1000, 2, [None, None], title) \
                  .getPDResult(matrices, [a1, a2], 'normal', noise)

            counts      = res["metrics"]["vecteur_de_comptes"]
            proportions = counts / n_runs
            x = np.arange(proportions.shape[0])

            # Génération automatique des labels pour vecteur_de_comptes
            n_codes = proportions.shape[1]
            k = int(np.sqrt(n_codes))
            if k*k == n_codes:
                labels = [f"({i},{j})" for i in range(k) for j in range(k)]
            else:
                labels = [f"pair {i}" for i in range(n_codes)]

            for code in range(n_codes):
                ax.plot(
                    x, proportions[:, code],
                    label=labels[code],
                    linewidth=1
                )

            ax.set_title(f"{a1} vs {a2}")
            sns.despine(ax=ax, trim=True)

        # axes partagés : on ajoute supxlabel et supylabel
        fig.supxlabel("Round (t)")
        fig.supylabel("Proportion over 500 runs")

        # légende globale en bas
        handles, labs = axes[1].get_legend_handles_labels()
        fig.legend(
            handles, labs,
            loc='lower center',
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02)
        )

        # ajustement final & sauvegarde
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        filename = f"Workshop/Figure/Figure v3/proportions_{game_name}_noise{noise[1]:.2f}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")

