import json

DG = [[1,0],
      [0,1]]

SG = [[2,0],
      [0,0.5]]

PARETO = [[1,0,0],
          [0,0.2,0],
          [0,0,1]]

#TODO - valeur de k à déterminer?
PARETO_P = [[1,0,-3],
            [0,0.2,0],
            -3,0,1]

PDA = [[3,5],
      [0,1]]
PDB = [[3,0],
      [5,1]]

PG_3= [[10 , 0, -3 ],
       [0.0, 2, 0  ],
       [-3 , 0, 10 ]]

matrices_dict = {
    "DG": [DG, DG],
    "SG":  [SG, SG],
    "PARETO": [PARETO, PARETO],
    "PARETO_P": [PARETO_P, PARETO_P],
    "PD":  [PDA, PDB],
    "PG_3": [PG_3, PG_3]
}

algos = ["UCB", "TS", "KLUCB"]
# constantes associées
# TODO - pour l'instant mettre 0 pour les algos qui ont pas besoin de la var "constant"
algo_consts = {"UCB": 2, "TS": 0, "KLUCB": 0}
noise_choices = [0, 0.1, 1]

horizon = 1000
rounds  = 1000
n_agents = 2

# 2) Construction de la liste des configs
pd_configs = []
for name, matrices in matrices_dict.items():
    for noise in noise_choices:
        for algo in algos:
            const = algo_consts[algo]
            # titre : <MATRICE>_<ALGO1><C1>_<ALGO2><C2>_N<NOISE>
            # on utilise :g pour éviter les .0 superflus
            title = f"{name}_{algo}{const:g}_{algo}{const:g}_N{noise:g}_{rounds}"
            pd_configs.append({
                "horizon": horizon,
                "rounds": rounds,
                "n_agents": n_agents,
                "const": [const, const],
                "noise_dist": "normal",
                "noise_params": [0.0, noise],
                "matrices": matrices,
                "algos": [algo, algo],
                "title": title
            })
        consts = [algo_consts["UCB"], algo_consts["TS"]]
        pd_configs.append({
            "horizon": horizon,
            "rounds": rounds,
            "n_agents": n_agents,
            "const": [algo_consts["UCB"], algo_consts["TS"]],
            "noise_dist": "normal",
            "noise_params": [0.0, noise],
            "matrices": matrices,
            "algos": ["UCB", "TS"],
            "title": f"{name}_UCB{consts[0]:g}_TS{consts[1]:g}_N{noise:g}_{rounds}"
        })

payload = {"pd": pd_configs}

with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)