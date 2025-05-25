import json

SG = [[2,0],
      [0,1]]

PG2 = [[2,0],
      [0,2]]

PDA = [[3,5],
      [0,1]]
PDB = [[3,0],
      [5,1]]

PG_3= [[10 , 0, -3 ],
       [0.0, 2, 0  ],
       [-3 , 0, 10 ]]

PG_9= [[10 , 0, -3 ],
       [0.0, 2, 0  ],
       [-3 , 0, 10 ]]

CG  =  [[11 ,-30, 0 ],
       [-30, 7 , 0 ],
       [0  , 0 , 5 ]]

matrices_dict = {
    "SG":  [SG, SG],
    "PG2": [PG2, PG2],
    "PD":  [PDA, PDB],
    "PG_3": [PG_3, PG_3],
    "PG_9": [PG_9, PG_9],
    "CG":  [CG, CG],
}


algos = ["UCB", "TUCB", "TS"]
# constantes associées
algo_consts = {"UCB": 2, "TUCB": 1.5, "TS": 1}
noise_choices = [0, 0.1, 1]

horizon = 1000
rounds  = 1000
n_agents = 2

# 2) Construction de la liste des configs
pd_configs = []
for name, matrices in matrices_dict.items():
    for algo1 in algos:
        for algo2 in algos:
            consts = [algo_consts[algo1], algo_consts[algo2]]
            for noise in noise_choices:
                # titre : <MATRICE>_<ALGO1><C1>_<ALGO2><C2>_N<NOISE>
                # on utilise :g pour éviter les .0 superflus
                title = f"{name}_{algo1}{consts[0]:g}_{algo2}{consts[1]:g}_N{noise:g}_{rounds}"
                pd_configs.append({
                    "horizon": horizon,
                    "rounds": rounds,
                    "n_agents": n_agents,
                    "const": consts,
                    "noise_dist": "normal",
                    "noise_params": [0.0, noise],
                    "matrices": matrices,
                    "algos": [algo1, algo2],
                    "title": title
                })

payload = {"pd": pd_configs}

print(json.dumps(payload, indent=2))
with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)