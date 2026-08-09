"""
Structure Learning Methods Example

Demonstrates hill-climbing, tabu search, and Max-Min Hill-Climbing (mmhc),
and compares the learned DAGs on a bundled dataset.
"""

import numpy as np
import pandas as pd

from mdmp import MDM, load_dataset

np.random.seed(42)

data_df = load_dataset("covid_regional_timeseries")
_, n_nodes = data_df.shape

print("=" * 60)
print("Structure Learning Methods Comparison")
print("=" * 60)
print(f"\nData shape: {data_df.shape}")
print(f"Variables: {list(data_df.columns)}")

methods = [
    ("hc", "Hill-Climbing"),
    ("tabu", "Tabu Search"),
    ("mmhc", "Max-Min Hill-Climbing"),
]
models = {}

for key, label in methods:
    print("\n" + "=" * 60)
    print(f"Method: {label} ({key})")
    print("=" * 60)
    model = MDM(data_df, method=key, nbf=15, verbose=False)
    models[key] = model
    adj = pd.DataFrame(model.adj_mat, index=model.node_names, columns=model.node_names)
    print("\nLearned Adjacency Matrix:")
    print(adj)
    print(f"Total edges: {int(np.sum(model.adj_mat))}")

print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
for key, label in methods:
    print(f"  {label}: {int(np.sum(models[key].adj_mat))} edges")

# Highlight edges that differ between hc and mmhc
diff = models["hc"].adj_mat - models["mmhc"].adj_mat
if np.any(diff != 0):
    print("\nEdges that differ (hc vs mmhc):")
    for i in range(n_nodes):
        for j in range(n_nodes):
            if diff[i, j] == 0:
                continue
            edge = f"{models['hc'].node_names[i]} -> {models['hc'].node_names[j]}"
            if diff[i, j] > 0:
                print(f"  {edge}: only in hill-climbing")
            else:
                print(f"  {edge}: only in Max-Min Hill-Climbing")
else:
    print("\nhc and mmhc found the same structure!")

print("\nDiscount Factors (hill-climbing):")
for name, df_val in zip(models["hc"].node_names, models["hc"].DF["DF_hat"]):
    print(f"  {name}: {df_val:.4f}")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
