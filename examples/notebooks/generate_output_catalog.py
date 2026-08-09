"""Generate newer catalog plots into examples/notebooks/output/.

Writes only the missing numbered assets (11–14). Does not overwrite 01–10.
"""

from __future__ import annotations

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
    plot_dendrogram,
    plot_group_embedding,
    plot_projection,
)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

print("Fitting MDM on mdmr_test_data...")
model = MDM(load_dataset("mdmr_test_data"), method="hc", nbf=15, verbose=False)


def _save(fig, name: str) -> None:
    path = OUTPUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


print("11. plot_anomalies...")
_save(plot_anomalies(model, series=0, ci_level=0.95), "11_plot_anomalies.png")

print("Building GS cohort...")
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

print("12. plot_dendrogram...")
fig_d, ax_d = plt.subplots(figsize=(7, 4))
plot_dendrogram(dist, ax=ax_d)
_save(fig_d, "12_plot_dendrogram.png")

print("13. plot_projection...")
fig_p, ax_p = plt.subplots(figsize=(6, 5))
plot_projection(dist, technique="mds", n_clusters=2, ax=ax_p)
_save(fig_p, "13_plot_projection.png")

print("14. plot_group_embedding...")
_save(
    plot_group_embedding(dist, technique="mds", n_clusters=2),
    "14_plot_group_embedding.png",
)

print(f"Done. Catalog directory: {OUTPUT}")
