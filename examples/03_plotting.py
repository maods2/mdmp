"""
Plotting Functions Example

This example demonstrates all plotting functions available in MDMP:
1. plot_dag() - DAG structure visualization (graph and heatmap)
2. plot_arcs() - Dynamic parameters over time
3. plot_marginal() - Marginal posterior for a target node
4. plot_stream() - Parent contributions to a child node
5. plot_idag() - Animated heatmap (optional, creates GIF)

Note: Some plots require matplotlib display backend. Adjust as needed.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
from mdmp import MDM, plot_dag, plot_arcs, plot_marginal, plot_stream

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic time series data
data_df = pd.read_csv("./data/example_dag.csv")
_, N = data_df.shape

print("=" * 60)
print("Plotting Functions Example")
print("=" * 60)

# Fit MDM model
print("\nFitting MDM model...")
model = MDM(data_df, method="hc", nbf=15, verbose=False)

# Create output directory for plots
import os
output_dir = "plot_examples"
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving plots to '{output_dir}/' directory...")

# 1. Plot DAG - Graph view
print("\n1. Plotting DAG (graph view)...")
fig1 = plot_dag(model, plot_type="graph", figsize=(10, 8))
fig1.savefig(f"{output_dir}/dag_graph.png", dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"   Saved: {output_dir}/dag_graph.png")

# 2. Plot DAG - Heatmap view
print("\n2. Plotting DAG (heatmap view)...")
fig2 = plot_dag(model, plot_type="heatmap", figsize=(8, 8))
fig2.savefig(f"{output_dir}/dag_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"   Saved: {output_dir}/dag_heatmap.png")

# 3. Plot dynamic parameters (connections)
print("\n3. Plotting dynamic parameters (connections)...")
fig3 = plot_arcs(model, plot_type="connections", distribution="filt", ci_level=0.95)
fig3.savefig(f"{output_dir}/dynamic_parameters.png", dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"   Saved: {output_dir}/dynamic_parameters.png")

# 4. Plot dynamic parameters (intercepts)
print("\n4. Plotting dynamic parameters (intercepts)...")
fig4 = plot_arcs(model, plot_type="intercepts", distribution="filt")
fig4.savefig(f"{output_dir}/intercepts.png", dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f"   Saved: {output_dir}/intercepts.png")

# 5. Plot marginal posterior for first node
print("\n5. Plotting marginal posterior for node 'A'...")
fig5 = plot_marginal(model, target_node=0, distribution="filt")
fig5.savefig(f"{output_dir}/marginal_posterior.png", dpi=150, bbox_inches='tight')
plt.close(fig5)
print(f"   Saved: {output_dir}/marginal_posterior.png")

# 6. Plot marginal posterior (smoothed)
print("\n6. Plotting marginal posterior (smoothed) for node 'A'...")
fig6 = plot_marginal(model, target_node=0, distribution="smoo")
fig6.savefig(f"{output_dir}/marginal_posterior_smoothed.png", dpi=150, bbox_inches='tight')
plt.close(fig6)
print(f"   Saved: {output_dir}/marginal_posterior_smoothed.png")

# 7. Plot stream (parent contributions) - only if node has parents
print("\n7. Plotting parent contributions (stream plot)...")
# Find a node with parents
node_with_parents = None
for i in range(N):
    if np.sum(model.adj_mat[:, i]) > 0:  # Has at least one parent
        node_with_parents = i
        break

if node_with_parents is not None:
    node_name = model.node_names[node_with_parents]
    fig7 = plot_stream(model, child_node=node_with_parents, distribution="filt")
    fig7.savefig(f"{output_dir}/stream_plot_{node_name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig7)
    print(f"   Saved: {output_dir}/stream_plot_{node_name}.png (for node {node_name})")
else:
    print("   Skipped: No nodes with parents found")

# 8. Plot animated heatmap (creates GIF)
print("\n8. Creating animated heatmap (this may take a moment)...")
try:
    from mdmp.plotting import plot_idag
    anim = plot_idag(
        model,
        output_gif=f"{output_dir}/animated_heatmap.gif",
        fps=5,
        distribution="filt"
    )
    print(f"   Saved: {output_dir}/animated_heatmap.gif")
except Exception as e:
    print(f"   Note: Animated plot requires additional dependencies: {e}")

print("\n" + "=" * 60)
print("All plots saved successfully!")
print(f"Check the '{output_dir}/' directory for output files.")
print("=" * 60)
