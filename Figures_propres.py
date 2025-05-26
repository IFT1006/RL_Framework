import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# 1) Importation des données
cwd = Path().resolve()
csv_dir  = cwd / "Workshop" / "Data"
csv_path = csv_dir / "A_combined.csv"   
df = pd.read_csv(csv_path)


# 2) Choix des paramètres des courbes
plt.rcParams.update({
    "text.usetex":        True,               
    "font.family":        "serif",            
    "font.serif":         ["Times New Roman"],
    "figure.dpi":         300,                
    "axes.titlesize":     14,                 
    "axes.labelsize":     12,                 
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
})
sns.set_theme(style="whitegrid", palette="colorblind")

# 3) Sans bruit
configs = ["UCB2_UCB2", "TS1_TS1", "UCB2_TS1"]

# Liste de marqueurs à faire tourner
markers = ["o", "s", "D", "^", "v", "P"]


def figure_propres(configs, legendes, save, game, bruit, agent):
    plt.figure(figsize=(5, 5), dpi=300)

    for idx, config in enumerate(configs):
        mean_cumul = df[f"{game}_{config}_N{bruit}_1000_mean_cum_regret_agent_{agent}"]
        std_cumul  = df[f"{game}_{config}_N{bruit}_1000_std_cum_regret_agent_{agent}"]

        plt.plot(
            df['step'],
            mean_cumul,
            label=legendes[idx],
            linewidth=0.5,
            marker=markers[idx],         # ← marqueur différent
            markevery=30,                # un marqueur tous les 20 points
            markersize=2                 # taille du marqueur
        )
        plt.fill_between(
            df['step'],
            mean_cumul,
            mean_cumul + std_cumul,
            alpha=0.3
        )
    # 2) Ajustement de la légende : hors de la zone de tracé en haut à droite
    plt.legend(loc="lower right", frameon=False)
    # 3) Labels et grille
    plt.xlabel("Time Step (t)")
    plt.ylabel("Cumulative Regret $R(t)$")
    sns.despine(trim=True)
    plt.grid(False)
    plt.tight_layout()
    # 4) Sauvegarde et affichage
    plt.savefig(f"Workshop/Figure/Figures propres/{save}.pdf")
    plt.show()

for agent in [0,1]:
    for bruit in ["0", "0.1", "1"]:
        configs = ["UCB2_UCB2", "TS1_TS1", "UCB2_TS1"]
        legendes = ["UCB vs UCB", "TS vs TS", "UCB vs TS"]
        game="SG"
        figure_propres(configs, legendes, f"{game}_analyse_bruit_{bruit}_agent_{agent}", game, bruit, agent)
