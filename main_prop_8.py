import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from execute import Execute


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


matrice_CG =  [[1,0,0.75],[0,0.9,0.85],[0.75,0.75,0.85]]
matrices = [np.array(matrice_CG), np.array(matrice_CG)]


for noise in [[0.0, 0.0],[0.0, 0.1],[0.0, 1.0]]:

    n_runs = 50

    # Listes des paires à comparer
    algo_pairs = [["UCB", "UCB"],["KLUCB", "KLUCB"],["TS", "TS"],["UCB", "KLUCB"],["UCB", "TS"]]

    # Création des subplots (2 lignes, 1 colonne)
    fig, axes = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(6, 8), constrained_layout=True, )

    for ax, algo_pair in zip(axes, algo_pairs):
        # Récupérer les résultats
        title = f"{algo_pair[0]}_vs_{algo_pair[1]}_{noise}"
        res = Execute(n_runs, 100, 2, [None, None], title) \
            .getPDResult(matrices, algo_pair, 'normal', noise)

        # Extraire le vecteur de comptes et convertir en proportions
        counts = res["metrics"]["vecteur_de_comptes"]  # shape (1000, 4)
        print(counts.shape)
        proportions = counts / n_runs                   # shape (1000, 4)
        print(proportions)

        # Tracer chaque paire
        x = np.arange(proportions.shape[0])

        n = proportions.shape[1]

        # 1) Détecte la taille de la grille
        k = int(np.sqrt(n))
        if k * k == n:
            labels = [f"({i},{j})" for i in range(k) for j in range(k)]
        else:
            # fallback basique
            labels = [f"pair {i}" for i in range(n)]

        for code in range(n):
            ax.plot(
                x,
                proportions[:, code],
                label=labels[code],
                linewidth=1,
            )

        # Titres et axes
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0, y=0)
        ax.set_title(f"{algo_pair[0]} X {algo_pair[1]}")
        sns.despine(ax=ax, trim=True)

    # Label x commun en bas

    fig.supylabel("Proportion over 500 runs")
    axes[-1].set_xlabel("Round (t)")

    # 1) Récupère handles/labels depuis un des axes
    handles, labels = axes[1].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=4,                # adapte selon ton nombre de paires
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)  # 0.5=center, -0.02 un peu en dessous
    )

    fig.subplots_adjust(bottom=0.30)  # augmente si besoin
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(
        "Workshop/Figure/Figure v3/proportions_{noise}.pdf",
        dpi=300, bbox_inches="tight"
    )
    plt.show()