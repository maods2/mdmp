"""
Individual Structure (IS) Aggregation Example

Builds a small set of subject adjacency matrices, aggregates them into a
consensus DAG with ``aggregate_individual_structures``, and plots the result.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdmp import aggregate_individual_structures, plot_dag

np.random.seed(42)

print("=" * 60)
print("Individual Structure (IS) Aggregation Example")
print("=" * 60)

# Four-node synthetic subject DAGs (adj[i, j] = 1 => parent i -> child j)
node_names = ["A", "B", "C", "D"]
subjects = [
    np.array(  # A→B, B→C
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int
    ),
    np.array(  # A→B, B→C (same majority path)
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int
    ),
    np.array(  # A→C, C→D
        [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=int
    ),
    np.array(  # A→B, A→C
        [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int
    ),
    np.array(  # B→C, C→D
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=int
    ),
    np.array(  # A→B only
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int
    ),
]

print(f"\nSubjects: {len(subjects)}, nodes: {node_names}")
for i, adj in enumerate(subjects):
    print(f"  subject {i}: {int(adj.sum())} edges")

print("\nAggregating with tau=0.5 (mc_n_samples=0)...")
result = aggregate_individual_structures(
    subjects,
    tau=0.5,
    node_names=node_names,
    mc_n_samples=0,
)

print("\nConsensus adjacency:")
print(result.adj_mat)
print(f"Total consensus edges: {int(np.sum(result.adj_mat))}")
if getattr(result, "metadata", None):
    print(f"Metadata keys: {sorted(result.metadata.keys())}")

output_dir = Path(__file__).resolve().parent / "plot_examples"
output_dir.mkdir(parents=True, exist_ok=True)
fig = plot_dag(result, plot_type="graph", figsize=(8, 6), layout_seed=3)
out = output_dir / "is_consensus_dag.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved consensus DAG: {out}")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
