import os,json
import argparse
from execute import Execute
from utils import *
import numpy as np
from tqdm import tqdm

def load_configs(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = path if os.path.isabs(path) else os.path.join(script_dir, path)

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path!r}")
    with open(cfg_path, "r") as f:
        data = json.load(f)
    return data.get("pd", [])

def run_pd_experiments(configs):
    for cfg in tqdm(configs):
        # 1) reconstruire les matrices
        matrices = [np.array(m) for m in cfg["matrices"]]
        # 2) extraire la config de bruit
        dist   = cfg.get("noise_dist",   "uniform")
        params = tuple(cfg.get("noise_params"))
        # 3) lancer l'expérience
        res = Execute(
            cfg["rounds"], cfg["horizon"],
            cfg["n_agents"], cfg["const"], cfg["title"]
        ).getPDResult(
            matrices,
            cfg["algos"],
            noise_dist=dist,
            noise_params=params,
        )
        plot_mean_std(res, cfg["title"])

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
    pd_configs = load_configs(args.config)

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
