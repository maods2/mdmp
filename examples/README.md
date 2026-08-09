# MDMP Examples

Runnable scripts live in this directory; Jupyter demos are under
[`notebooks/`](notebooks/).

## Scripts

| Script | What it shows |
|---|---|
| `01_basic_usage.py` | Load a bundled dataset, fit MDM, inspect adj / DF / Filt / Smoo |
| `02_structure_learning.py` | Compare `hc`, `tabu`, and `mmhc` |
| `03_plotting.py` | All public plot APIs (DAG, arcs, marginal, stream, idag, anomalies, GS plots) |
| `04_vts_usage.py` | Virtual Typical Subject (`compute_vts`) with list / 3D / DataFrame input |
| `05_is_aggregation.py` | Individual Structure consensus via `aggregate_individual_structures` |
| `06_gs_distance.py` | Per-subject fit → `compute_mdm_distance` → clusters / dendrogram |

```bash
python examples/01_basic_usage.py
python examples/02_structure_learning.py
python examples/03_plotting.py
python examples/04_vts_usage.py
python examples/05_is_aggregation.py
python examples/06_gs_distance.py
```

Plotting scripts write figures under `examples/plot_examples/` (non-interactive Agg backend).

## Notebooks

Canonical tours and case studies:

- [`notebooks/01-mdmp-library-demo.ipynb`](notebooks/01-mdmp-library-demo.ipynb) — end-to-end retail demo
- [`notebooks/05-is-aggregation.ipynb`](notebooks/05-is-aggregation.ipynb) — IS aggregation
- [`notebooks/08-gs-distance-projection.ipynb`](notebooks/08-gs-distance-projection.ipynb) — GS distance / projection
- [`notebooks/04-is-vs-vts-multi-individual.ipynb`](notebooks/04-is-vs-vts-multi-individual.ipynb) — IS vs VTS
- [`notebooks/09-gs-clusters-then-vts-is.ipynb`](notebooks/09-gs-clusters-then-vts-is.ipynb) — GS clusters then VTS/IS

Retail helpers and CSV ship beside the notebooks:
`notebooks/retail_helpers.py`, `notebooks/data/MDM_retail_dataset.csv`.

Static plot catalog assets live in `notebooks/output/` (numbered `01`–`14`).
Regenerate the newer anomaly/GS assets with:

```bash
python examples/notebooks/generate_output_catalog.py
```

## Prerequisites

```bash
pip install -e .
```

Examples use bundled datasets via `load_dataset` (or synthetic arrays). Plotting
examples need matplotlib; animation may need extra imageio/pillow deps.

## Notes

- Fixed random seeds (typically 42) for reproducibility
- Small cohorts / short series so scripts finish quickly
- Adjust `nbf`, cohort size, and methods for your own experiments

## Further reading

- Main package overview: [`../README.md`](../README.md)
- API details: function / class docstrings in `mdmp/`
