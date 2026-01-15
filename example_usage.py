"""
Test MDMP package with COVID regional time series data.

This script mirrors test_covid.R functionality and compares Python implementation
results with the R package (mdmr) when possible.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from mdmp import MDM, plot_dag, plot_arcs, plot_marginal
import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Change to script directory
os.chdir(script_dir)

# Check if we're in the right location
if not (script_dir / "mdmp").exists() and not (script_dir.parent / "covid_regional_timeseries.csv").exists():
    # Try parent directory
    parent_dir = script_dir.parent
    if (parent_dir / "covid_regional_timeseries.csv").exists():
        os.chdir(parent_dir)
    else:
        print(f"Warning: Could not find covid_regional_timeseries.csv")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Parent directory: {parent_dir}")

#######################################################
# LOAD CSV DATA
#######################################################
print("\n" + "=" * 60)
print("Loading COVID regional time series data...")
print("=" * 60)

# Get root directory (parent of mdmp)
root_dir = Path(os.getcwd()).parent if "mdmp" in os.getcwd() else Path(os.getcwd())
csv_file = root_dir / "covid_regional_timeseries.csv"

# Also try current directory
if not csv_file.exists():
    csv_file = Path("covid_regional_timeseries.csv")
    if not csv_file.exists():
        csv_file = script_dir.parent / "covid_regional_timeseries.csv"

if not csv_file.exists():
    print(f"\nERROR: CSV file not found: {csv_file}")
    print(f"Current directory: {os.getcwd()}")
    print("Please ensure covid_regional_timeseries.csv exists in the parent directory.")
    sys.exit(1)

# Load the CSV data
print(f"Reading CSV file: {csv_file}")
ts_data = pd.read_csv(csv_file)

print("Data loaded successfully!")
print(f"Time series dimensions: {ts_data.shape[0]} x {ts_data.shape[1]}")
print(f"Variables: {', '.join(ts_data.columns)}")
print("\nFirst few rows:")
print(ts_data.head())

# Convert to DataFrame for MDM function
ts_df = ts_data.copy()

#######################################################
# RUN MDM ANALYSIS
#######################################################
print("\n" + "=" * 60)
print("Running MDM analysis with hill-climbing...")
print("=" * 60)

# Run MDM analysis with hill-climbing
result = MDM(data=ts_df, method="hc", nbf=15, verbose=True)

print("\nMDM analysis completed successfully!")
print("=" * 60)
print("Result summary:")
print(f"  - Type: {type(result).__name__}")
print(f"  - Components: {', '.join([k for k in result.__dict__.keys() if not k.startswith('_')])}")
print(f"  - Adjacency matrix dimensions: {result.adj_mat.shape}")
print(f"\nAdjacency matrix:")
print(result.adj_mat)

print("\nDiscount factors (DF_hat):")
for i, df_val in enumerate(result.DF['DF_hat']):
    node_name = result.node_names[i] if hasattr(result, 'node_names') else f"Node {i}"
    print(f"  {node_name}: {df_val:.6f}")

# Print adjacency matrix in a more readable format
print("\nAdjacency matrix (formatted):")
adj_df = pd.DataFrame(
    result.adj_mat,
    index=result.node_names,
    columns=result.node_names
)
print(adj_df)

# Count edges
num_edges = np.sum(result.adj_mat)
print(f"\nTotal number of edges: {num_edges}")
if num_edges > 0:
    print("\nEdges (parent -> child):")
    for i in range(len(result.node_names)):
        for j in range(len(result.node_names)):
            if result.adj_mat[i, j] == 1:
                print(f"  {result.node_names[i]} -> {result.node_names[j]}")

#######################################################
# PLOT MDM RESULTS
#######################################################
print("\n" + "=" * 60)
print("Plotting MDM results...")
print("=" * 60)

# Create output directory for plots
output_dir = Path("mdmp_output")
output_dir.mkdir(exist_ok=True)

# Plot 1: DAG Structure (Graph)
print("\nPlotting DAG structure (graph)...")
try:
    fig1 = plot_dag(result, plot_type="graph")
    fig1.savefig(output_dir / "dag_structure.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'dag_structure.png'}")
    plt.close(fig1)
except Exception as e:
    print(f"   Error plotting DAG graph: {e}")

# Plot 2: DAG Structure (Heatmap)
print("\nPlotting DAG structure (heatmap)...")
try:
    fig2 = plot_dag(result, plot_type="heatmap")
    fig2.savefig(output_dir / "dag_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'dag_heatmap.png'}")
    plt.close(fig2)
except Exception as e:
    print(f"   Error plotting DAG heatmap: {e}")

# Plot 3: Dynamic Parameters (Connections)
print("\nPlotting dynamic parameters (connections)...")
try:
    fig3 = plot_arcs(result, plot_type="connections", distribution="filt", ci_level=0.95)
    fig3.savefig(output_dir / "dynamic_parameters.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'dynamic_parameters.png'}")
    plt.close(fig3)
except Exception as e:
    print(f"   Error plotting dynamic parameters: {e}")

# Plot 4: Marginal posterior for first node
print("\nPlotting marginal posterior for node 0...")
try:
    fig4 = plot_marginal(result, target_node=0, distribution="filt")
    fig4.savefig(output_dir / "marginal_posterior.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'marginal_posterior.png'}")
    plt.close(fig4)
except Exception as e:
    print(f"   Error plotting marginal posterior: {e}")

#######################################################
# SAVE RESULTS FOR COMPARISON
#######################################################
print("\n" + "=" * 60)
print("Saving results for comparison with R implementation...")
print("=" * 60)

# Save adjacency matrix
adj_output_file = output_dir / "adjacency_matrix.csv"
adj_df.to_csv(adj_output_file)
print(f"   Saved adjacency matrix: {adj_output_file}")

# Save discount factors
df_output_file = output_dir / "discount_factors.csv"
df_df = pd.DataFrame({
    'Node': result.node_names,
    'DF_hat': result.DF['DF_hat']
})
df_df.to_csv(df_output_file, index=False)
print(f"   Saved discount factors: {df_output_file}")

# Save summary statistics
summary_file = output_dir / "summary.txt"
with open(summary_file, 'w') as f:
    f.write("MDMP Analysis Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Data dimensions: {ts_df.shape[0]} x {ts_df.shape[1]}\n")
    f.write(f"Variables: {', '.join(ts_df.columns)}\n")
    f.write(f"Number of edges: {num_edges}\n")
    f.write(f"\nAdjacency Matrix:\n")
    f.write(adj_df.to_string())
    f.write(f"\n\nDiscount Factors:\n")
    f.write(df_df.to_string(index=False))
    f.write(f"\n")

print(f"   Saved summary: {summary_file}")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
print(f"\nResults saved to: {output_dir.absolute()}")
print("\nTo compare with R implementation:")
print("1. Run: Rscript mdmr/test_covid.R")
print("2. Compare adjacency matrices and discount factors")
print("3. Note: Due to random search in hill-climbing, results may differ slightly")

