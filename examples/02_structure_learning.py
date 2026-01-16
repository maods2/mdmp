"""
Structure Learning Methods Example

This example demonstrates:
1. Different structure learning methods (hill-climbing vs tabu search)
2. How to use each method
3. Comparing learned structures
"""

import numpy as np
import pandas as pd
from mdmp import MDM

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic time series data
data_df = pd.read_csv("./data/example_dag.csv")
_, N = data_df.shape

print("=" * 60)
print("Structure Learning Methods Comparison")
print("=" * 60)
print(f"\nData shape: {data_df.shape}")
print(f"Variables: {list(data_df.columns)}")

# Method 1: Hill-climbing (default)
print("\n" + "=" * 60)
print("Method 1: Hill-Climbing")
print("=" * 60)
model_hc = MDM(data_df, method="hc", nbf=15, verbose=False)

print("\nLearned Adjacency Matrix:")
adj_hc = pd.DataFrame(
    model_hc.adj_mat,
    index=model_hc.node_names,
    columns=model_hc.node_names
)
print(adj_hc)
print(f"Total edges: {np.sum(model_hc.adj_mat)}")

# Method 2: Tabu Search
print("\n" + "=" * 60)
print("Method 2: Tabu Search")
print("=" * 60)
model_tabu = MDM(data_df, method="tabu", nbf=15, verbose=False, max_iter=20)

print("\nLearned Adjacency Matrix:")
adj_tabu = pd.DataFrame(
    model_tabu.adj_mat,
    index=model_tabu.node_names,
    columns=model_tabu.node_names
)
print(adj_tabu)
print(f"Total edges: {np.sum(model_tabu.adj_mat)}")

# Compare structures
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"\nHill-climbing edges: {np.sum(model_hc.adj_mat)}")
print(f"Tabu search edges: {np.sum(model_tabu.adj_mat)}")

# Find differences
diff = model_hc.adj_mat - model_tabu.adj_mat
if np.any(diff != 0):
    print("\nEdges that differ:")
    for i in range(N):
        for j in range(N):
            if diff[i, j] != 0:
                if diff[i, j] > 0:
                    print(f"  {model_hc.node_names[i]} -> {model_hc.node_names[j]}: "
                          f"only in hill-climbing")
                else:
                    print(f"  {model_hc.node_names[i]} -> {model_hc.node_names[j]}: "
                          f"only in tabu search")
else:
    print("\nBoth methods found the same structure!")

# Discount factors comparison
print("\nDiscount Factors Comparison:")
print("\nHill-climbing:")
for name, df_val in zip(model_hc.node_names, model_hc.DF['DF_hat']):
    print(f"  {name}: {df_val:.4f}")

print("\nTabu search:")
for name, df_val in zip(model_tabu.node_names, model_tabu.DF['DF_hat']):
    print(f"  {name}: {df_val:.4f}")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
