import os
import glob
import json
import numpy as np
import pandas as pd

base_folder = "/home/hyunho_RCI/datasets/gr00t-rl/forge/gear_mesh" # TODO: Edit This!!!
parquet_folder = os.path.join(base_folder, "data/chunk-000")
stats_json_path = os.path.join(base_folder, "meta/stats.json")

os.makedirs(os.path.dirname(stats_json_path), exist_ok=True)

parquet_files = glob.glob(os.path.join(parquet_folder, "**", "*.parquet"), recursive=True)
if len(parquet_files) == 0:
    print(f"No parquet files found in {parquet_folder}")
    exit()

print(f"Found {len(parquet_files)} parquet files.")

all_data = []
for file in parquet_files:
    df = pd.read_parquet(file)
    all_data.append(df)
df_all = pd.concat(all_data, ignore_index=True)

stats = {}

for col in df_all.columns:
    sample_val = df_all[col].iloc[0]
    if isinstance(sample_val, (list, np.ndarray)):
        arr = np.stack(df_all[col])
        stats[col] = {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist()
        }
    elif np.issubdtype(type(sample_val), np.number):
        arr = df_all[col].values.astype(np.float64)
        stats[col] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q01": float(np.quantile(arr, 0.01)),
            "q99": float(np.quantile(arr, 0.99))
        }
    else:
        continue

with open(stats_json_path, "w") as f:
    json.dump(stats, f, indent=4)
print(f"Saved stats.json to {stats_json_path}")
