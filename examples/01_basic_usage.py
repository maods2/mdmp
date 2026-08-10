"""
Basic MDM Usage Example

Demonstrates:
1. Loading a bundled dataset
2. Fitting an MDM model
3. Inspecting adjacency, discount factors, and filter/smooth outputs
"""

import numpy as np
import pandas as pd

from mdmp import MDM, load_dataset

np.random.seed(42)

data_df = load_dataset("mdmr_test_data")
_, n_nodes = data_df.shape

print("=" * 60)
print("Basic MDM Usage Example")
print("=" * 60)
print(f"\nData shape: {data_df.shape}")
print(f"Variables: {list(data_df.columns)}")
print("\nFirst few rows:")
print(data_df.head())

print("\n" + "=" * 60)
print("Fitting MDM model...")
print("=" * 60)
model = MDM(data_df, method="hc", nbf=15, verbose=True)

print("\n" + "=" * 60)
print("Model Results")
print("=" * 60)

print("\nAdjacency Matrix (learned structure):")
adj_df = pd.DataFrame(
    model.adj_mat,
    index=model.node_names,
    columns=model.node_names,
)
print(adj_df)

num_edges = int(np.sum(model.adj_mat))
print(f"\nTotal edges: {num_edges}")

print("\nSelected Discount Factors:")
for name, df_val in zip(model.node_names, model.DF["DF_hat"]):
    print(f"  {name}: {df_val:.4f}")

print("\nFiltered Parameters Summary:")
print(f"  Number of nodes: {len(model.Filt['mt'])}")
for i, name in enumerate(model.node_names):
    mt_shape = model.Filt["mt"][i].shape
    print(f"  {name}: {mt_shape[0]} parameters, {mt_shape[1]} time points")

print("\nSmoothed Parameters Summary:")
print(f"  Number of nodes: {len(model.Smoo['smt'])}")
for i, name in enumerate(model.node_names):
    smt_shape = model.Smoo["smt"][i].shape
    print(f"  {name}: {smt_shape[0]} parameters, {smt_shape[1]} time points")

print("\n" + "=" * 60)
print("Model Summary:")
print("=" * 60)
print(repr(model))
print(f"(n_nodes={n_nodes})")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
