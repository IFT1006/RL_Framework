import pandas as pd
from pathlib import Path
from functools import reduce
from tqdm import tqdm

data_dir = Path("Workshop/Data")
csv_files = list(data_dir.glob("*.csv"))

dfs = []
for csv_path in tqdm(csv_files):
    title = csv_path.stem   # ex. "CG_TS1_UCB2_N0.1_100"
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        col: f"{title}_{col}"
        for col in df.columns
        if col != "step"
    })

    # 3) on garde 'step' en commun puis tout le reste préfixé
    dfs.append(df)

# 4) merge séquentiel sur 'step'
combined = reduce(
    lambda left, right: pd.merge(left, right, on="step", how="outer"),
    dfs
)

# 5) écriture du CSV combiné
output_path = data_dir / "A_combined.csv"
combined.to_csv(output_path, index=False)
