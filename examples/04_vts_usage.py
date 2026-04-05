"""
Virtual Typical Subject (VTS) Usage Example

This example demonstrates:
1. Creating multi-subject time series data
2. Computing VTS via concatenation and mean-based approaches
3. Using VTS output with MDM for structure learning
"""

import numpy as np
import pandas as pd

from mdmp import MDM, compute_vts, load_dataset

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("Virtual Typical Subject (VTS) Usage Example")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Create multi-subject data
# ---------------------------------------------------------------------------
# Option A: Split a single dataset into "subjects" (simulated multi-subject)
# Option B: Use list of arrays with variable lengths
# Option C: Use 3D array (I x k x N)

# Load base data and split into 3 "subjects" for demonstration
data_single = load_dataset("covid_regional_timeseries")
T, N = data_single.shape
print(f"\nBase data shape: {data_single.shape} (T={T}, N={N} variables)")

# Split into 3 subjects (each gets ~1/3 of time points)
split_points = [0, T // 3, 2 * T // 3, T]
multi_subject_list = [
    data_single.values[split_points[i] : split_points[i + 1], :]
    for i in range(3)
]
print(f"Multi-subject (list): 3 subjects, shapes {[a.shape for a in multi_subject_list]}")

# Also create aligned 3D array (I x k x N) for mean-based
k = min(a.shape[0] for a in multi_subject_list)
aligned_3d = np.array([a[:k, :] for a in multi_subject_list])
print(f"Aligned 3D array shape: {aligned_3d.shape} (I x k x N)")

# ---------------------------------------------------------------------------
# 2. Compute VTS with both methods
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Computing VTS")
print("=" * 60)

# Mean-based VTS (subject-level mean, then group mean)
result_mean = compute_vts(aligned_3d, method="mean")
print(f"\nMean-based VTS:")
print(f"  Shape: {result_mean.vts_data.shape}")
print(f"  Method: {result_mean.method}")
print(f"  N subjects: {result_mean.n_subjects}")

# Concatenation-based VTS (concatenate along time, return series for MDM)
result_concat = compute_vts(multi_subject_list, method="concatenation")
print(f"\nConcatenation-based VTS (return_series=True):")
print(f"  Shape: {result_concat.vts_data.shape}")
print(f"  Method: {result_concat.method}")

# Concatenation with return_series=False (global mean over time)
result_concat_summary = compute_vts(
    multi_subject_list, method="concatenation", return_series=False
)
print(f"\nConcatenation-based VTS (return_series=False, summary):")
print(f"  Shape: {result_concat_summary.vts_data.shape}")

# ---------------------------------------------------------------------------
# 3. Use VTS with MDM
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Fitting MDM on VTS")
print("=" * 60)

# Fit MDM on mean-based VTS (typical subject structure)
print("\nFitting MDM on mean-based VTS...")
model_vts = MDM(result_mean.vts_data, method="hc", verbose=False)
print(f"  Learned structure: {np.sum(model_vts.adj_mat)} edges")

# Fit MDM on concatenation VTS (pooled data structure)
print("\nFitting MDM on concatenation VTS...")
model_pooled = MDM(result_concat.vts_data, method="hc", verbose=False)
print(f"  Learned structure: {np.sum(model_pooled.adj_mat)} edges")

# ---------------------------------------------------------------------------
# 4. DataFrame input (long format with subject_id)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. DataFrame Input (long format)")
print("=" * 60)

# Build long-format DataFrame (subject_id + variable columns)
rows = []
for i, arr in enumerate(multi_subject_list):
    for t in range(arr.shape[0]):
        row = {"subject_id": i}
        for j, name in enumerate(data_single.columns):
            row[name] = arr[t, j]
        rows.append(row)
df_long = pd.DataFrame(rows)
print(f"Long-format DataFrame: {df_long.shape}")
print(f"  Columns: {list(df_long.columns)}")

result_df = compute_vts(df_long, method="mean")
print(f"  VTS from DataFrame: shape {result_df.vts_data.shape}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
