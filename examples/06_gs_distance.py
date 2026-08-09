"""
Group-Structure (GS) Distance Example

Fits per-subject MDMs, computes pairwise MDM distance, clusters subjects,
and saves a dendrogram (Agg backend).
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdmp import (
    compute_mdm_distance,
    fit_individual_structures,
    plot_dendrogram,
    plot_group_embedding,
)

np.random.seed(42)


def _make_cohort(n_per_group: int = 2, n_time: int = 60, n_nodes: int = 3, seed: int = 7):
    """Two latent wiring patterns (same idea as the distance tests)."""
    rng = np.random.default_rng(seed)
    subjects = []
    labels = []

    def gen(kind: str) -> np.ndarray:
        e = rng.normal(0, 1, size=(n_time, n_nodes))
        x = np.zeros((n_time, n_nodes))
        x[:, 0] = e[:, 0]
        if kind == "A":
            x[:, 1] = 0.8 * x[:, 0] + 0.4 * e[:, 1]
            x[:, 2] = 0.8 * x[:, 1] + 0.4 * e[:, 2]
        else:
            x[:, 2] = 0.8 * x[:, 0] + 0.4 * e[:, 2]
            x[:, 1] = 0.8 * x[:, 2] + 0.4 * e[:, 1]
        return x

    for _ in range(n_per_group):
        subjects.append(gen("A"))
        labels.append(0)
    for _ in range(n_per_group):
        subjects.append(gen("B"))
        labels.append(1)
    return subjects, labels


print("=" * 60)
print("Group-Structure (GS) Distance Example")
print("=" * 60)

subjects, true_groups = _make_cohort()
subject_ids = [f"s{i}" for i in range(len(subjects))]
print(f"\nCohort: {len(subjects)} subjects, shapes {[a.shape for a in subjects]}")
print(f"Latent groups (for reference): {true_groups}")

print("\n1. fit_individual_structures...")
inds = fit_individual_structures(
    subjects,
    method="hc",
    nbf=10,
    subject_ids=subject_ids,
    verbose=False,
)
print(f"  Fitted {len(inds)} MDMs; edges={[int(m.adj_mat.sum()) for m in inds]}")

print("\n2. compute_mdm_distance...")
dist = compute_mdm_distance(inds, nbf=10, verbose=False)
print("  Distance matrix:")
print(np.round(dist.matrix, 3))

print("\n3. cluster_labels (k=2)...")
labels = dist.cluster_labels(2)
print(f"  Cluster labels: {labels.tolist()}")

output_dir = Path(__file__).resolve().parent / "plot_examples"
output_dir.mkdir(parents=True, exist_ok=True)

fig_d, ax_d = plt.subplots(figsize=(7, 4))
plot_dendrogram(dist, ax=ax_d)
dendro_path = output_dir / "gs_dendrogram.png"
fig_d.savefig(dendro_path, dpi=150, bbox_inches="tight")
plt.close(fig_d)
print(f"\nSaved: {dendro_path}")

fig_e = plot_group_embedding(dist, technique="mds", n_clusters=2)
embed_path = output_dir / "gs_group_embedding.png"
fig_e.savefig(embed_path, dpi=150, bbox_inches="tight")
plt.close(fig_e)
print(f"Saved: {embed_path}")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
