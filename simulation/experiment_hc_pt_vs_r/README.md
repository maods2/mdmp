# MDMp vs MDMr — Hill-Climbing Reproducibility Experiment

Compare **Python `mdmp`** and **R `mdmr`** hill-climbing (`method="hc"`) on identical
simulated time series. Data is generated once in Python; both engines read the same
CSVs and we test whether they recover the **same adjacency matrix** across 300
replications per DAG.

## Pinned parameters

| Parameter | Value |
|-----------|-------|
| Scenario | `W=0.01`, `V=100.0`, `T=200` (Michel TCC setting) |
| DAGs | `3var` (chain), `5var` (Figura 6) |
| Replications | 300 per DAG |
| `base_seed` | 1564 (individual `i` uses `1564 + i`) |
| `nbf` (burn-in) | **15** |
| `delta` grid | **`np.linspace(0.5, 1.0, 51)`** — 51 values from 0.5 to 1.0 (avoids float drift past 1.0) |
| Method | `"hc"` only |
| Adjacency orientation | **row = parent, col = child** |

Python call:

```python
MDM(data, method="hc", nbf=15, delta=np.linspace(0.5, 1.0, 51), verbose=False)
```

R call (see notebook 03 header for `?mdm` verification):

```r
CDELT <- seq(0.5, 1.0, by = 0.01)
mdm(data_input = data, method = "hc", nbf = 15, delta = CDELT)
```

(`mdmr` expects the argument name **`delta`**, not `CDELT`.)

## Sample data generation

All time series are simulated in Python only. Both `mdmp` and `mdmr` read the same
CSV files from `data/`, so there is a single random-number source and no
engine-specific data drift. **Do not simulate in R** — the CSVs are the sole
source of truth (see also the note under [Run order](#run-order)).

Implementation lives in [`../simulate_dags.py`](../simulate_dags.py) (Capítulo 5
simulation study). Notebook **`01_generate_data.ipynb`** prepends the parent
`simulation/` directory to `sys.path`, imports `run_simulations`, and writes
into this folder's `data/`:

```python
sys.path.insert(0, os.path.abspath(".."))
from simulate_dags import run_simulations

run_simulations(
    output_dir="data/",
    base_seed=1564,
    n_individuals=300,
    w_values=(0.01,),
    v_values=(100.0,),
    t_values=(200,),
)
```

### DAG structures

Two fixed topologies are generated. Adjacency matrices use **row = parent,
col = child** (a `1` at `[i, j]` means `Yi → Yj`).

**3-variable chain (`3var`, Figura 5):** `Y1 → Y2 → Y3`.

**5-variable graph (`5var`, Figura 6):** `Y1` is the root with edges
`Y1 → Y2`, `Y1 → Y3`; `Y4` has parents `Y2` and `Y3`; `Y5` has parent `Y2`.

### DLM simulation model

Each replication is a multivariate dynamic linear model (DLM) time series of
length `T`. At every time step `t`:

1. **State evolution (random walk):** `θ_t = θ_{t−1} + w_t` with
   `w_t ~ N(0, W)` component-wise. The state vector `θ` holds one intercept
   coefficient per node plus one coefficient per incoming parent edge (e.g.
   `p = 5` parameters for the 3-var chain, `p = 10` for the 5-var graph).

2. **Observation equations:** each node `Y_j(t)` equals its intercept term
   from `θ`, plus parent contributions (regression on parent values at time
   `t`), plus observation noise `v_j(t) ~ N(0, V)`.

   For the 3-var chain:
   - `Y1(t) = θ_1(t) + v_1(t)`
   - `Y2(t) = θ_2(t) + θ_3(t)·Y1(t) + v_2(t)`
   - `Y3(t) = θ_4(t) + θ_5(t)·Y2(t) + v_3(t)`

   The 5-var graph follows the same pattern with the parent sets defined by
   the Figura 6 topology (see docstrings in `simulate_dag_5var`).

### Fixed scenario and replications

This experiment pins a single variance/sample-size setting from the Michel TCC
configuration:

| Parameter | Value | Role |
|-----------|-------|------|
| `W` | `0.01` | System (state) variance |
| `V` | `100.0` | Observational variance |
| `T` | `200` | Number of time points per series |

For each DAG, **300 independent replications** share the same topology and
`(W, V, T)` but differ in their random draws. Replication `k` (filename suffix
`ind{k}`, for `k = 1…300`) uses RNG seed `base_seed + (k − 1)` with
`base_seed = 1564` (seeds `1564` through `1863`).

### Output file naming

For each DAG and `(W, V, T)` combination, `run_simulations` writes:

- **Time series:** `dag_{3var|5var}_W{W}_V{V}_T{T}_ind{k}.csv` — columns
  `Y1`, `Y2`, … (no index column); one file per replication.
- **Ground-truth adjacency:** `dag_{3var|5var}_W{W}_V{V}_T{T}_true_adjacency.csv`
  — same for all replications of that DAG (topology is fixed).

With the pinned scenario, examples are
`dag_3var_W0.01_V100.0_T200_ind42.csv` and
`dag_5var_W0.01_V100.0_T200_true_adjacency.csv`.

## Run order

1. **`01_generate_data.ipynb`** (Python) — simulate 300×2 datasets via `simulate_dags.run_simulations`
2. **`02_run_python_hc.ipynb`** (Python) — fit `mdmp` HC on every dataset
3. **`03_run_r_hc.ipynb`** (R, IRkernel) — fit `mdmr` HC on the **same** CSVs
4. **`04_compare.ipynb`** (Python) — zero-matrix adjacency diff test + metrics/time summary

**Important:** never simulate in R. The CSV files in `data/` are the single source of truth.

## Dependencies

**Python:** `mdmp`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, Jupyter

**R:** `mdmr`, `bnlearn`, `reshape2`, `dplyr`, IRkernel

Install IRkernel and register the R kernel before running notebook 03.

## Runtime warning

This experiment runs **600 HC fits per engine** (300 per DAG × 2 DAGs). Expect
hours of wall-clock time depending on hardware. Notebooks 02 and 03 write results
at the end; re-run individual notebooks if interrupted.

## Output files (`data/`)

Generated by notebook 01:

- `dag_{3var|5var}_W0.01_V100.0_T200_ind{1..300}.csv`
- `dag_{3var|5var}_W0.01_V100.0_T200_true_adjacency.csv`

Generated by notebook 02:

- `python_adjacency.csv` — long format: `dag, dataset_id, node_from, node_to, edge`
- `python_metrics.csv` — long format: `dag, dataset_id, metric, value`

Generated by notebook 03 (same schema):

- `r_adjacency.csv`
- `r_metrics.csv`

Generated by notebook 04 (in-memory / displayed):

- Per-dataset adjacency diff (`R_adj − Python_adj`)
- Summary tables per DAG

## Acceptance criterion

For each `(dag, dataset_id)`, `R_adj − Python_adj` should be an all-zeros matrix.
Disagreements may arise from HC score ties or floating-point LPL differences — inspect
rather than treat as automatic failures.
