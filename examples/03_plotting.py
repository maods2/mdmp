"""
Plotting Functions Example

Covers the public plotting entry points:
  plot_dag, plot_arcs, plot_marginal, plot_stream, plot_idag,
  plot_anomalies, plot_dendrogram, plot_projection, plot_group_embedding

Figures are saved under examples/plot_examples/ (Agg backend).
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdmp import (
    MDM,
    compute_mdm_distance,
    fit_individual_structures,
    load_dataset,
    plot_anomalies,
    plot_arcs,
    plot_dag,
    plot_dendrogram,
    plot_group_embedding,
    plot_idag,
    plot_marginal,
    plot_projection,
    plot_stream,
)

np.random.seed(42)

data_df = load_dataset("mdmr_test_data")
_, n_nodes = data_df.shape

print("=" * 60)
print("Plotting Functions Example")
print("=" * 60)

print("\nFitting MDM model...")
model = MDM(data_df, method="hc", nbf=15, verbose=False)

output_dir = Path(__file__).resolve().parent / "plot_examples"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\nSaving plots to '{output_dir}/' ...")


def _save(fig, name: str) -> None:
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path.name}")


print("\n1. plot_dag (graph)...")
_save(plot_dag(model, plot_type="graph", figsize=(10, 8), layout_seed=5), "dag_graph.png")

print("\n2. plot_dag (heatmap)...")
_save(plot_dag(model, plot_type="heatmap", figsize=(8, 8)), "dag_heatmap.png")

print("\n3. plot_arcs (connections)...")
_save(
    plot_arcs(model, plot_type="connections", distribution="filt", ci_level=0.95),
    "dynamic_parameters.png",
)

print("\n4. plot_arcs (intercepts)...")
_save(plot_arcs(model, plot_type="intercepts", distribution="filt"), "intercepts.png")

print("\n5. plot_marginal (filtered)...")
_save(plot_marginal(model, target_node=0, distribution="filt"), "marginal_posterior.png")

print("\n6. plot_marginal (smoothed)...")
_save(
    plot_marginal(model, target_node=0, distribution="smoo"),
    "marginal_posterior_smoothed.png",
)

print("\n7. plot_stream (parent contributions)...")
node_with_parents = next(
    (i for i in range(n_nodes) if np.sum(model.adj_mat[:, i]) > 0),
    None,
)
if node_with_parents is not None:
    node_name = model.node_names[node_with_parents]
    _save(
        plot_stream(model, child_node=node_with_parents, distribution="filt"),
        f"stream_plot_{node_name}.png",
    )
else:
    print("   Skipped: no nodes with parents")

print("\n8. plot_anomalies...")
_save(plot_anomalies(model, series=0, ci_level=0.95), "anomalies.png")

print("\n9. plot_idag (animated heatmap)...")
try:
    plot_idag(
        model,
        output_gif=str(output_dir / "animated_heatmap.gif"),
        fps=5,
        distribution="filt",
    )
    print("   Saved: animated_heatmap.gif")
except Exception as exc:  # pragma: no cover - optional deps / backend
    print(f"   Note: animated plot skipped ({exc})")

# Tiny multi-subject cohort for GS plots
print("\n10. GS cohort plots (dendrogram / projection / embedding)...")
rng = np.random.default_rng(0)
subjects = []
for _ in range(4):
    e = rng.normal(size=(50, 3))
    x = np.zeros_like(e)
    x[:, 0] = e[:, 0]
    x[:, 1] = 0.6 * x[:, 0] + e[:, 1]
    x[:, 2] = e[:, 2]
    subjects.append(x)

inds = fit_individual_structures(subjects, method="hc", nbf=10, verbose=False)
dist = compute_mdm_distance(inds, nbf=10, verbose=False)

fig_d, ax_d = plt.subplots(figsize=(7, 4))
plot_dendrogram(dist, ax=ax_d)
_save(fig_d, "dendrogram.png")

fig_p, ax_p = plt.subplots(figsize=(6, 5))
plot_projection(dist, technique="mds", n_clusters=2, ax=ax_p)
_save(fig_p, "projection.png")

_save(plot_group_embedding(dist, technique="mds", n_clusters=2), "group_embedding.png")

print("\n" + "=" * 60)
print("All plots saved successfully!")
print(f"Check '{output_dir}' for output files.")
print("=" * 60)
