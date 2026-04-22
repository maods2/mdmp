"""
Structure Learning Methods Example

This example demonstrates:
1. Different structure learning methods (hill-climbing vs Max-Min Hill-Climbing)
2. How to use each method
3. Comparing learned structures
"""

import numpy as np
import pandas as pd

from mdmp import MDM, load_dataset

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic time series data
data_df = load_dataset("covid_regional_timeseries")
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

# Method 2: Max-Min Hill-Climbing
print("\n" + "=" * 60)
print("Method 2: Max-Min Hill-Climbing")
print("=" * 60)
model_mmhc = MDM(data_df, method="mmhc", nbf=15, verbose=False)

print("\nLearned Adjacency Matrix:")
adj_mmhc = pd.DataFrame(
    model_mmhc.adj_mat,
    index=model_mmhc.node_names,
    columns=model_mmhc.node_names
)
print(adj_mmhc)
print(f"Total edges: {np.sum(model_mmhc.adj_mat)}")

# Compare structures
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"\nHill-climbing edges: {np.sum(model_hc.adj_mat)}")
print(f"Max-Min Hill-Climbing edges: {np.sum(model_mmhc.adj_mat)}")

# Find differences
diff = model_hc.adj_mat - model_mmhc.adj_mat
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
                          f"only in Max-Min Hill-Climbing")
else:
    print("\nBoth methods found the same structure!")

# Discount factors comparison
print("\nDiscount Factors Comparison:")
print("\nHill-climbing:")
for name, df_val in zip(model_hc.node_names, model_hc.DF['DF_hat']):
    print(f"  {name}: {df_val:.4f}")

print("\nMax-Min Hill-Climbing:")
for name, df_val in zip(model_mmhc.node_names, model_mmhc.DF['DF_hat']):
    print(f"  {name}: {df_val:.4f}")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
