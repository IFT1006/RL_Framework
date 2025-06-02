import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

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


def figure_propres_regret(configs, legendes, save, game, bruit, agent):
    for placement_legende in ["lower right", "upper left"]:
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

        plt.legend(loc=placement_legende, frameon=False)
        # 3) Labels et grille
        plt.xlabel("Time Step (t)")
        plt.ylabel("Cumulative Regret $R(t)$")
        sns.despine(trim=True)
        plt.grid(False)
        plt.tight_layout()
        # 4) Sauvegarde et affichage
        plt.savefig(f"Workshop/Figure/Figures propres/{save}_{placement_legende}.pdf")
        plt.close()


for game in tqdm(['SG', 'PG2', 'PD', 'PG_3', 'CG']):
    agent = 0
    for bruit in ["0", "0.1", "1"]:
        configs = ["UCB2_UCB2", "UCB2_TS1", "UCB2_TUCB1.5"]
        legendes = ["UCB vs UCB", "UCB vs TS", "UCB vs TUCB"]
        figure_propres_regret(configs, legendes, f"analyse_de_UCB_{game}_bruit_{bruit}_agent_{agent}", game, bruit, agent)
    for bruit in ["0", "0.1", "1"]:
        configs = ["TS1_UCB2", "TS1_TS1", "TS1_TUCB1.5"]
        legendes = ["UCB vs UCB", "UCB vs TS", "UCB vs TUCB"]
        figure_propres_regret(configs, legendes, f"analyse_de_TS_{game}_bruit_{bruit}_agent_{agent}", game, bruit, agent)
    for bruit in ["0", "0.1", "1"]:
        configs = ["TUCB1.5_UCB2", "TUCB1.5_TS1", "TUCB1.5_TUCB1.5"]
        legendes = ["TUCB vs UCB", "TUCB vs TS", "TUCB vs TUCB"]
        figure_propres_regret(configs, legendes, f"analyse_de_TUCB_{game}_bruit_{bruit}_agent_{agent}", game, bruit, agent)
    for agent in [0,1]:
        for bruit in ["0", "0.1", "1"]:
            configs = ["UCB2_UCB2", "TS1_TS1", "UCB2_TS1"]
            legendes = ["UCB vs UCB", "TS vs TS", "UCB vs TS"]
            figure_propres_regret(configs, legendes, f"analyse_sur_le bruit_game_{game}_bruit_{bruit}_agent_{agent}", game, bruit, agent)



def to_cartesian(pa, pb, pc):
    """
    Convertit (pa,pb,pc) avec pa+pb+pc=1 en (x,y) dans un triangle équilatéral.
    """
    # Triangle de côté 1 
    x = 0.5*(2*pb + pc) / (pa + pb + pc)
    y = (np.sqrt(3)/2) * pc / (pa + pb + pc)
    return x, y

def plot_ternary_overlay_manual(df, game, config, noise, agents=(0,1),
                                actions=(0,1,2), colors=("C0","C1"), markers=("o","s"),
                                save_name="ternary_manual"):
    steps = df['step'].values  # pour éventuellement colorer au fil du step

    fig, ax = plt.subplots(figsize=(6,6))
    # Triangle de base
    corners = np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2],[0,0]])
    ax.plot(corners[:,0], corners[:,1], 'k-', lw=1.0)

    # Étiquettes
    ax.text(-0.02,-0.02, "a", fontsize=12)
    ax.text(1.02,-0.02, "b", fontsize=12)
    ax.text(0.48,0.9,   "c", fontsize=12)

    # Trace chaque agent
    for idx, agent in enumerate(agents):
        pa = df[f"{game}_{config}_N{noise}_1000_prop_action_0_agent_{agent}"].values
        pb = df[f"{game}_{config}_N{noise}_1000_prop_action_1_agent_{agent}"].values
        pc = df[f"{game}_{config}_N{noise}_1000_prop_action_2_agent_{agent}"].values

        xs, ys = to_cartesian(pa, pb, pc)
        ax.scatter(
            xs, ys,
            c=colors[idx],
            marker=markers[idx],
            label=f"Agent {agent}",
            s=30,
            alpha=0.7
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"{game} {config} (Overlay Agents)")
    plt.tight_layout()
    plt.savefig(f"{save_name}_manual.pdf")
    plt.show()


plot_ternary_overlay_manual(df, "PG_3", "UCB2_UCB2","0.1")