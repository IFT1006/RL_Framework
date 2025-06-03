import numpy as np
from execute import Execute
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd


def run_one_game_experiments(game_name, matrices, noise_levels, algos, rounds=500, horizon=1000, n_agents=2):
    """
    Pour un jeu donné, fait tourner toutes les combinaisons d'algorithmes et de niveaux de bruit,
    puis renvoie un dict {titre: result} où 'result' est le retour de Execute.getPDResult.
    """
    results = {}
    for noise in tqdm(noise_levels):
        for algo_pair in algos:
            title = f"{algo_pair[0]}×{algo_pair[1]}_{noise[1]}"
            res = Execute(
                rounds,
                horizon,
                n_agents,
                [None] * n_agents,
                game_name
            ).getPDResult(
                matrices,
                algo_pair,
                'normal',
                noise
            )
            results[title] = res
    return results


def plot_results(game_name, results, noise_levels, algos,y_axes, save_folder="Workshop/Figure"):
    sharey = False
    fig, axes = plt.subplots(1, 3, sharey=sharey, figsize=(15, 4))
    for ax, noise in zip(axes, noise_levels):
        for algo_paired in algos:
            title = f"{algo_paired[0]}×{algo_paired[1]}_{noise[1]}"
            mean = results[title]['metrics']['mean_cum_regret']['agent_0']
            std = results[title]['metrics']['std_cum_regret']['agent_0']
            x = np.arange(len(mean))
            ax.plot(
                x,
                mean,
                label=f"{algo_paired[0]}$\\times${algo_paired[1]}",
                linewidth=1,
            )
            ax.fill_between(
                x,
                mean,
                mean + std,
                alpha=0.2
            )
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0, y=0)
        ax.set_title(f"$\\sigma_{{noise}}={noise[1]}$")
        ax.set_xlabel("Round (t)")
        sns.despine(ax=ax, trim=True)

    # 3) Labels communs
    for i in y_axes:
        axes[0].set_ylim(axes[i].get_ylim())
        axes[1].set_ylim(axes[i].get_ylim())
        axes[2].set_ylim(axes[i].get_ylim())

    sns.set_style("whitegrid")
    for ax in axes:
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", frameon=False)

    # only the leftmost gets the y‑axis label
    axes[0].set_ylabel("Mean cumulative regret $R(t)$")
    axes[0].set_ylabel("Mean cumulative regret $R(t)$")
    plt.tight_layout()
    fig.savefig(f"{save_folder}/{game_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_results_action_mode(game_name, results, noise_levels, algos, save_folder="Workshop/Figure"):
    for noise in noise_levels:
        table_data = {}
        for algo_pair in algos:
            title = f"{algo_pair[0]}×{algo_pair[1]}_{noise[1]}"
            data = results[title]['metrics']["vecteur_de_proportion_mode"]
            table_data[f"{algo_pair[0]}×{algo_pair[1]}"] = data

        # Create DataFrame and save as CSV
        df = pd.DataFrame(table_data)
        filename = f"{save_folder}/{game_name}_action_mode_{noise[1]}.csv"
        df.to_csv(filename)


def plot_results_exploration(game_name, results, noise_levels, algos,y_axes, save_folder="Workshop/Figure"):
    sharey = False
    fig, axes = plt.subplots(1, 3, sharey=sharey, figsize=(15, 4))
    for ax, noise in zip(axes, noise_levels):
        for algo_paired in algos:
            title = f"{algo_paired[0]}×{algo_paired[1]}_{noise[1]}"
            print('title',title)
            mean = results[title]['metrics']['mean_exploration_conjointe']
            x = np.arange(len(mean))
            ax.plot(
                x,
                mean,
                label=f"{algo_paired[0]}$\\times${algo_paired[1]}",
                linewidth=1,
            )

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0, y=0)
        ax.set_title(f"$\\sigma_{{noise}}={noise[1]}$")
        ax.set_xlabel("Round (t)")
        sns.despine(ax=ax, trim=True)

    # 3) Labels communs
    for i in y_axes:
        axes[0].set_ylim(axes[i].get_ylim())
        axes[1].set_ylim(axes[i].get_ylim())
        axes[2].set_ylim(axes[i].get_ylim())

    sns.set_style("whitegrid")
    for ax in axes:
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", frameon=False)

    # only the leftmost gets the y‑axis label
    axes[0].set_ylabel("Conjoint exploration rate")
    plt.tight_layout()
    fig.savefig(f"{save_folder}/{game_name}_exploration.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


#
def plot_results_action(game_name, results, noise_levels, algos, n_runs, save_folder="Workshop/Figure"):
    for noise in noise_levels:
        # Création d’une nouvelle figure pour ce (jeu, bruit)
        fig, axes = plt.subplots(
            len(algos), 1,
            sharex=True, sharey=True,
            figsize=(6, 8),
            constrained_layout=True,
            dpi=300
        )

        for ax, (a1, a2) in zip(axes, algos):
            title = f"{a1}×{a2}_{noise[1]}"
            counts = results[title]["metrics"]["vecteur_de_comptes"]
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
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.margins(x=0, y=0)
            ax.set_title(f"{a1}$\\times${a2}")
            sns.despine(ax=ax, trim=True)

        # axes partagés : on ajoute supxlabel et supylabel
        fig.supxlabel("Round (t)")
        fig.supylabel("Proportion over 500 runs")

        # légende globale en bas
        label = []
        handles, labs = axes[1].get_legend_handles_labels()
        # incrémenter par 1 car action est indexé sur 1 et non 0
        label.extend(labs)
        updated_label = []
        for lab in label:
            x_str, y_str = lab.strip('()').split(',')
            x = int(x_str.strip()) + 1
            y = int(y_str.strip()) + 1
            updated_label.append(f'({x},{y})')

        fig.legend(
            handles, updated_label,
            loc='lower center',
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.1)
        )

        # sauvegarde
        filename = f"{save_folder}/proportions_{game_name}_noise{noise[1]:.2f}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    np.random.seed(43)

    # Configuration générale des styles
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "figure.dpi": 300,
        "axes.titlesize": 5,
        "axes.labelsize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
    })
    sns.set_theme(style="whitegrid", palette="colorblind")

    # Dictionnaire des jeux avec leurs matrices respectives
    games = {
        "PG_wp": [[np.array([[1, 0.1, 0], [0.1, 0.28, 0.1], [0, 0.1, 1]]),
                  np.array([[1, 0.1, 0], [0.1, 0.28, 0.1], [0, 0.1, 1]])], [2,1,2]],
        "PG": [[np.array([[1, 0, 0], [0, 0.2, 0], [0, 0, 1]]),
               np.array([[1, 0, 0], [0, 0.2, 0], [0, 0, 1]])], [0,0,2]],
        "PD": [[np.array([[0.6, 0], [1, 0.4]]),
               np.array([[0.6, 0], [1, 0.4]]).T], [1,1,2]],
        "SG": [[np.array([[1, 0], [0, 0.5]]),
               np.array([[1, 0], [0, 0.5]])],[1,1,2]],
        "CG_no": [[np.array([[1, 0, 0.75], [0, 0.9, 0.85], [0.75, 0.75, 0.85]]),
                  np.array([[1, 0, 0.75], [0, 0.9, 0.85], [0.75, 0.75, 0.85]])],[2,2,2]]
    }

    # Paramètres communs à toutes les expériences
    noise_levels = [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0]]
    algo_pairs = [
        ["UCB", "UCB"],
        ["KLUCB", "KLUCB"],
        ["TS", "TS"],
        ["UCB", "KLUCB"],
        ["UCB", "TS"]
    ]

    for game in tqdm(games.keys()):
        if game == 'PD':
            results = run_one_game_experiments(game ,games[game][0], noise_levels, algo_pairs, rounds=50, horizon=100000, n_agents=2)
            plot_results(game, results, noise_levels, algo_pairs, games[game][1],save_folder="Workshop/figure_long")
            plot_results_action(game, results, noise_levels, algo_pairs, 500, save_folder="Workshop/figure_long")
            plot_results_exploration(game, results, noise_levels, algo_pairs,games[game][1] , save_folder="Workshop/figure_long")
            plot_results_action_mode(game, results, noise_levels, algo_pairs, save_folder="Workshop/figure_long")
