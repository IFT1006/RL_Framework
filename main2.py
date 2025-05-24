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
        res = Execute(
            cfg["horizon"], cfg["rounds"],
            cfg["n_agents"], cfg["noise_std"]
        ).getPDResult(
            [np.array(m) for m in cfg["matrices"]],
            cfg["algos"]
        )
        prop = np.array(res["prop"])
        printProp3(prop, cfg["title"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", "-c", type=str, required=True,
        help="Chemin vers le JSON d’expériences"
    )
    p.add_argument(
        "--which", choices=["bandit","pd","all"], default="all",
        help="Type d’expériences à lancer"
    )
    p.add_argument(
        "--exp-id", type=int, default=None,
        help="(optionnel) index 0‑based de l’expérience bandit à ne lancer que celle‑ci"
    )
    args = p.parse_args()

    bandit_configs, pd_configs = load_configs(args.config)

    if args.which in ("bandit","all"):
        # si exp-id fourni, on ne garde que cet index pour bandit
        to_run = (
            [bandit_configs[args.exp_id]]
            if args.exp_id is not None
            else bandit_configs
        )
        print(f"→ Lancement de {len(to_run)} expérience(s) Bandit …")
        run_bandit_experiments(to_run)

    if args.which in ("pd","all"):
        print(f"→ Lancement de {len(pd_configs)} expérience(s) PD …")
        run_pd_experiments(pd_configs)

if __name__ == "__main__":
    main()
