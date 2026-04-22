"""
Basic MDM Usage Example

This example demonstrates how to:
1. Create synthetic time series data
2. Fit an MDM model
3. Access and inspect the results (adjacency matrix, discount factors, parameters)
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
print("Basic MDM Usage Example")
print("=" * 60)
print(f"\nData shape: {data_df.shape}")
print(f"Variables: {list(data_df.columns)}")
print("\nFirst few rows:")
print(data_df.head())

# Fit MDM model with default parameters (hill-climbing)
print("\n" + "=" * 60)
print("Fitting MDM model...")
print("=" * 60)
model = MDM(data_df, method="hc", nbf=15, verbose=True)

# Access results
print("\n" + "=" * 60)
print("Model Results")
print("=" * 60)

# Adjacency matrix (learned DAG structure)
print("\nAdjacency Matrix (learned structure):")
adj_df = pd.DataFrame(
    model.adj_mat,
    index=model.node_names,
    columns=model.node_names
)
print(adj_df)

# Count edges
num_edges = np.sum(model.adj_mat)
print(f"\nTotal edges: {num_edges}")

# Discount factors
print("\nSelected Discount Factors:")
for name, df_val in zip(model.node_names, model.DF['DF_hat']):
    print(f"  {name}: {df_val:.4f}")

# Filtered parameters summary
print("\nFiltered Parameters Summary:")
print(f"  Number of nodes: {len(model.Filt['mt'])}")
for i, name in enumerate(model.node_names):
    mt_shape = model.Filt['mt'][i].shape
    print(f"  {name}: {mt_shape[0]} parameters, {mt_shape[1]} time points")

# Smoothed parameters summary
print("\nSmoothed Parameters Summary:")
print(f"  Number of nodes: {len(model.Smoo['smt'])}")
for i, name in enumerate(model.node_names):
    smt_shape = model.Smoo['smt'][i].shape
    print(f"  {name}: {smt_shape[0]} parameters, {smt_shape[1]} time points")

# Model representation
print("\n" + "=" * 60)
print("Model Summary:")
print("=" * 60)
print(repr(model))

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
