import os,json
import argparse
from execute import Execute
from utils import *
import numpy as np

def load_configs(path):
    # Determine absolute path of your script’s directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If the user passed a relative path, interpret it relative to script_dir
    cfg_path = path if os.path.isabs(path) else os.path.join(script_dir, path)

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path!r}")
    with open(cfg_path, "r") as f:
        data = json.load(f)
    return data.get("bandit", []), data.get("pd", [])

def run_bandit_experiments(configs):

    # Julien: À modifier car ça ne marche probablement plus
    for cfg in configs:
        res = Execute(
            cfg["horizon"], cfg["rounds"],
            cfg["n_agents"], cfg["noise_std"]
        ).getBanditResult(
            cfg["win_rate"], cfg["use_noise"], cfg["algo"]
        )
        avg = np.array(res["avg"])
        std = np.array(res["std"])
        printMean(avg, std, cfg["n_agents"], cfg["horizon"], cfg["algo"])
        printRuns(res["agents"], cfg["title"])

def run_pd_experiments(configs):
    for cfg in configs:
        # 1) reconstruire les matrices
        matrices = [np.array(m) for m in cfg["matrices"]]
        # 2) extraire la config de bruit
        dist   = cfg.get("noise_dist",   "uniform")
        params = tuple(cfg.get("noise_params"))
        # 3) lancer l'expérience
        res = Execute(
            cfg["horizon"], cfg["rounds"],
            cfg["n_agents"], cfg["const"], cfg["title"]
        ).getPDResult(
            matrices,
            cfg["algos"],
            noise_dist=dist,
            noise_params=params
        )
        print(res)
        # 4) afficher
        #printProp3(np.array(res["prop"]), cfg["title"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", "-c", type=str, required=True,
        help="Chemin vers le JSON d’expériences"
    )
    p.add_argument(
        "--exp-name", "-n", type=str, default=None,
        help="(optionnel) nom de l’expérience PD à lancer"
    )
    args = p.parse_args()

    # on ne récupère que les configs PD
    _, pd_configs = load_configs(args.config)

    # si on précise un exp-name, on filtre
    if args.exp_name:
        pd_configs = [
            cfg for cfg in pd_configs
            if cfg.get("title") == args.exp_name
        ]
        if not pd_configs:
            raise ValueError(
                f"Aucune expérience PD nommée '{args.exp_name}' dans {args.config}"
            )

    print(f"→ Lancement de {len(pd_configs)} expérience(s) PD …")
    run_pd_experiments(pd_configs)

if __name__ == "__main__":
    main()
