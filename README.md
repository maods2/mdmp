# MDMP: Bayesian Network Modeling for Dynamic Multivariate Time Series

**MDMP** is a new Python implementation of the Bayesian dynamic regression model
(MDM) for multivariate time series. It learns Bayesian network structure and
estimates time-varying parameters with Kalman filtering and smoothing
(hill-climbing, tabu search, and Max-Min Hill-Climbing).

## Features

- **Structure Learning**: Learn Bayesian network structures from multivariate time series (hill-climbing, tabu search, Max-Min Hill-Climbing)
- **Dynamic Parameter Estimation**: Estimate time-varying parameters using Kalman filtering and smoothing
- **Discount Factor Selection**: Automatically select optimal discount factors for each node
- **Parallel Processing**: Multiprocessing support for multi-core systems
- **Visualization**: Plot DAGs, dynamic parameters, marginal posteriors, streams, and animated heatmaps
- **Anomaly detection**: Flag observations outside the one-step predictive interval (`detect_anomalies` / `plot_anomalies`)
- **Virtual Typical Subject (VTS)**: Build a representative series from multi-subject data (mean, median, or concatenation), then fit MDM
- **Individual Structure (IS)**: Aggregate per-subject DAGs into a consensus structure (edge voting + acyclic repair)
- **Group-structure (GS) distance**: Pairwise MDM dissimilarities for clustering, embedding, and dendrograms

## MDM Algorithm Flow

The MDM algorithm follows this general workflow:

```mermaid
flowchart TD
    A[Multivariate Time Series Data] --> B[Structure Learning]
    B --> C{Select Method}
    C -->|hc| D[Hill-Climbing]
    C -->|tabu| E[Tabu Search]
    C -->|mmhc| F[Max-Min Hill-Climbing]
    
    D --> G[Initialize Structure]
    E --> G
    F --> G
    
    G --> H[Loop: Evaluate Candidate Structures]
    H --> I[For Each Node:<br/>Maximize LPL]
    I --> J[Optimize Discount Factor δ]
    J --> K[Compute Local Score<br/>Build Design Matrix<br/>Run DLM Filter<br/>Sum Log Predictive Likelihood]
    K --> L{Score Improved?}
    L -->|Yes| M[Update Structure<br/>Add/Remove/Reverse Edges]
    L -->|No| N{More Candidates?}
    M --> N
    N -->|Yes| H
    N -->|No| O[DAG Adjacency Matrix]
    
    O --> P[Select Discount Factors]
    P --> Q[For Each Node:<br/>Evaluate δ Values]
    Q --> R[Select δ that<br/>Maximizes LPL]
    R --> S[Discount Factors DF_hat]
    
    S --> T[Filter]
    T --> U[For Each Node:<br/>Run DLM Filter]
    U --> V[Build Design Matrix<br/>Extract Target Series<br/>Filter with DF_hat]
    V --> W[Filtered Parameters<br/>mt, Ct, Rt, nt, dt]
    
    W --> X[Smooth]
    X --> Y[For Each Node:<br/>Run DLM Smooth]
    Y --> Z[Smoothed Parameters<br/>smt, sCt, SE]
    
    style A fill:#e1f5ff
    style O fill:#fff4e1
    style S fill:#fff4e1
    style W fill:#e8f5e9
    style Z fill:#e8f5e9
```

**Key Steps:**

1. **Structure Learning**: Learn the DAG structure using a selected method (e.g., hill-climbing). The algorithm iteratively evaluates candidate structures by maximizing the log predictive likelihood (LPL) for each node.

2. **Discount Factor Selection**: For each node, evaluate different discount factors (δ) and select the one that maximizes the LPL.

3. **Filtering**: Run DLM filtering for each node using the learned structure and selected discount factors to obtain filtered dynamic parameters.

4. **Smoothing**: Run DLM smoothing on the filtered parameters to obtain smoothed estimates with reduced variance.

## Installation

### Install from PyPI

```bash
pip install mdmp
```

Optional extras:

```bash
pip install "mdmp[graphviz]"   # Graphviz DAG style (pydot + Graphviz binary)
pip install "mdmp[umap]"       # UMAP embeddings for GS projection
```

### Install from GitHub

```bash
pip install git+https://github.com/maods2/mdmp.git
pip install git+https://github.com/maods2/mdmp.git@v0.6.2
```

### Install from source

```bash
git clone https://github.com/maods2/mdmp.git
cd mdmp
pip install .
```

For editable installs, tests, and linting, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.2
- matplotlib >= 3.3.0
- networkx >= 2.6.0
- pgmpy >= 0.1.25
- tqdm >= 4.60.0

### Optional Dependencies

- **pydot** (+ Graphviz `dot` binary): `plot_dag(..., style="graphviz")` — `pip install mdmp[graphviz]`
- **umap-learn**: UMAP technique in group-structure projection — `pip install mdmp[umap]`
- **notears**: `method="notears"` structure learning (not on PyPI):

```bash
pip install git+https://github.com/xunzheng/notears.git
```

## Example Usage

The full end-to-end tour is the retail case study notebook
[`notebooks/01-mdmp-library-demo.ipynb`](notebooks/01-mdmp-library-demo.ipynb).
Below is a condensed walkthrough of the same flow: load retail sales, fit an MDM,
and plot the learned DAG. Group-analysis examples (VTS, IS, GS clustering) live in
`notebooks/` — see especially
[`notebooks/09-gs-clusters-then-vts-is.ipynb`](notebooks/09-gs-clusters-then-vts-is.ipynb).

### 1. Load retail sales and fit MDM

Helpers and the CSV ship under `notebooks/` (`retail_helpers.py`, `data/`).

```python
import sys
from pathlib import Path

import pandas as pd
from mdmp import MDM, plot_dag, plot_arcs

sys.path.insert(0, str(Path("notebooks").resolve()))
from retail_helpers import parse_retail_dataset, aggregate_by_level, DAG_LABELS

sales, hierarchy = parse_retail_dataset()
# Sum SKUs within each product line (C3), then learn structure
c3 = aggregate_by_level(sales, hierarchy, "type")
model = MDM(c3, method="hc", verbose=True, n_jobs=-1)
```

`model` holds the inferred DAG (`adj_mat`), filtering/smoothing estimates (`Filt`, `Smoo`),
and local scores. Structure-learning methods include `"hc"`, `"tabu"`, and `"mmhc"`.

### 2. Plot the DAG and dynamic arcs

```python
plot_dag(model, plot_type="graph", node_labels=DAG_LABELS)
plot_arcs(model, plot_type="connections", distribution="filt", ci_level=0.95)
```

For bundled sample series without the retail CSV, you can also use
`load_dataset("covid_regional_timeseries")` or `load_dataset("mdmr_test_data")`.

---

## Visualizations

The figures below illustrate the plotting API (examples use a fitted `MDM` object).

### DAG Structure

```python
plot_dag(model, plot_type="graph")
```

![DAG Graph](plot_examples/dag_graph.png)

> Displays the structure as a directed acyclic graph with nodes and directed edges.

- The `plot_type` argument selects the plot to display. Either `"graph"` for a directed acyclic graph with arrows or `"heatmap"` for a matrix view.

```python
plot_dag(model, plot_type="heatmap")
```

![DAG Heatmap](plot_examples/dag_heatmap.png)

- There are other arguments to personalize the graph output like `edge_color`, `node_color`, and `node_labels`.

### Arcs Over Time

Highlight which arcs were selected and their local scores:

```python
plot_arcs(model, plot_type="connections", distribution="filt", ci_level=0.95)
```

![Dynamic Parameters](plot_examples/dynamic_parameters.png)

> Helps identify dominant parent-child relationships.

- The `plot_type` argument selects the dynamic parameters time series that are displayed.
  - `"connections"`: Plots only dynamic edge parameters (e.g., V3→V1).
  - `"intercepts"`: Plots only intercepts (e.g., beta0 terms).
  - `"all"`: Plots both intercepts and connections.

```python
plot_arcs(model, plot_type="intercepts", distribution="filt")
```

![Intercepts](plot_examples/intercepts.png)

- The `distribution` argument defines whether to use filtered (`"filt"`) or smoothed (`"smoo"`) distributions. Default is `"filt"`.
- The `ci_level` argument establishes the width of the credible interval. Default is `0.95`.

### Dynamic Heatmap Animation

Animated heatmap of posterior estimates across time:

```python
plot_idag(
    mdm_object=model,
    output_gif="my_dynamic_model.gif",
    fps=10,
    width=6,
    height=6,
    dpi=100
)
```

![Animated Heatmap](plot_examples/animated_heatmap.gif)

> Each tile represents the magnitude of a dynamic parameter at each time step. This animation captures the temporally varying intensity of connections in the network structure.

- The `mdm_object` argument requires an object of class `MDM` as returned by `MDM()`.
- `output_gif` is the name of the output file. Must end with `.gif`. Default is `"mdm_dynamic.gif"`.
- `fps` selects the frames per second for the animation. Default is `10`.
- `width` (in inches) of each frame. Default is `6`.
- `height` (in inches) of each frame. Default is `6`.
- `dpi` is the resolution (dots per inch) for saved frames. Default is `100`.

### Stream Plot for a Node

Plot contribution of parents to the dynamic evolution of a target node:

```python
plot_stream(mdm_object=model, child_node=0, distribution="filt")
```

![Stream Plot](plot_examples/stream_plot_Y1.png)

> Useful for assessing how different parents dynamically influence a given node.

- The `mdm_object` argument requires an object of class `MDM` as returned by `MDM()`.
- The `child_node` argument is an integer index of the target node (i.e., the child whose parents' effects are shown).
- The `distribution` argument defines whether to use filtered (`"filt"`) or smoothed (`"smoo"`) posterior estimates. Default is `"filt"`.

### Marginal Posterior for a Node

Plot the marginal posterior means and confidence bands:

```python
plot_marginal(mdm_object=model, distribution="filt", target_node=0, scale_series=False)
```

![Marginal Posterior](plot_examples/marginal_posterior.png)

> Shows how the coefficients associated with a node evolve over time (filtered or smoothed).

```python
plot_marginal(mdm_object=model, distribution="smoo", target_node=0, scale_series=False)
```

![Marginal Posterior Smoothed](plot_examples/marginal_posterior_smoothed.png)

- The `mdm_object` argument requires an object of class `MDM` as returned by `MDM()`.
- The `target_node` argument is an integer index specifying the target node (child).
- The `distribution` argument defines whether to use filtered (`"filt"`) or smoothed (`"smoo"`) posterior estimates. Default is `"filt"`.
- The `scale_series` argument is a logical value. If `True`, all time series (observed and parental contributions) are standardized (mean zero, unit variance). Default is `False`.

### Anomaly detection (predictive interval)

Flag observations outside the MDM one-step Student-t predictive band and plot them:

```python
from mdmp import detect_anomalies, plot_anomalies

result = detect_anomalies(model, ci_level=0.95)          # all nodes
df = detect_anomalies(model, series="Y1", output="dataframe")
fig = plot_anomalies(model, series=0, ci_level=0.95)
```

- Criterion: anomaly if `y_t` lies outside `f_t ± t_{(1+ci)/2, n_t-1} √Q_t`.
- Score: absolute standardized forecast error `|ets|`.
- `plot_anomalies` calls `detect_anomalies` (no separate detection logic) and draws the observed polyline, predictive mean, interval band, and `x` markers on anomalies.

---

## Main Components

### MDM Class

The main `MDM` class coordinates structure learning, discount factor selection, filtering, and smoothing:

```python
model = MDM(
    data,              # Time series data (pd.DataFrame or np.ndarray)
    method="hc",       # Structure learning method: "hc", "tabu", "mmhc"
    nbf=15,            # Burn-in time point
    delta=None,        # Discount factor sequence (auto if None)
    verbose=True,      # Print progress and show progress bars
    n_jobs=-1          # Number of parallel jobs (-1 = all cores, 1 = serial, None = serial)
)
```

**Key Parameters:**
- `method`: Structure learning algorithm (`"hc"`, `"tabu"`, or `"mmhc"`)
- `verbose`: If `True`, shows progress messages, progress bars, and total processing time
- `n_jobs`: Parallel processing control:
  - `None` or `1`: Serial processing (default)
  - `-1`: Use all available CPU cores
  - `> 1`: Use that many parallel workers

**Attributes:**
- `adj_mat`: Adjacency matrix of learned DAG structure
- `data`: Original input data
- `DF`: Discount factor estimation results (contains `DF_hat` and `lpldet`)
- `Filt`: Filtered dynamic parameters (contains `mt`, `Ct`, `Rt`, `nt`, `dt`, `ft`, `Qt`, `ets`, `lpl`, `row_names`)
- `Smoo`: Smoothed dynamic parameters (contains `smt`, `sCt`, `SE`)
- `node_names`: Names of variables/nodes
- `verbose`: Whether verbose output is enabled
- `nbf`: Burn-in time point used

### Structure Learning Methods

By default, the `MDM()` function uses the hill-climbing algorithm to learn the structure of the Bayesian network. Other heuristic methods are also available.

Currently available methods:

- **`"hc"`**: **Hill-climbing** (default, fast) - Uses pgmpy's `HillClimbSearch` with custom MDM scoring function. Optimizes the log predictive likelihood per node.

- **`"mmhc"`**: **Max-Min Hill-Climbing** - First learns an undirected skeleton via MMPC (Max-Min Parents and Children), then orients edges using hill-climbing with custom MDM score.

- **`"tabu"`**: **Tabu search** - Uses pgmpy's HillClimbSearch with tabu search enabled (via `tabu_length` parameter). Supports `tabu_length` parameter (default: 100), `max_iter` (default: 1000000), `epsilon` (default: 0.0001), and other pgmpy HillClimbSearch parameters via `**kwargs`.

**Example with tabu search:**
```python
model = MDM(data, method="tabu", tabu_length=50, max_iter=1000, verbose=True)
```

**Registered but not yet implemented:**
- **`"ipa"`**: Integer Programming Approach (GOBNILP) — registered but currently raises `NotImplementedError`.

**Not yet implemented:**
- **`"h2pc"`**: H2PC algorithm
- **`"rsmax2"`**: RSMAX2 algorithm

**Note:** Implemented structure learners use `pgmpy` as the backend (included with `pip install mdmp`).

### Plotting Functions

1. **`plot_dag()`**: Plot DAG structure as graph or heatmap
2. **`plot_arcs()`**: Plot dynamic parameters over time
3. **`plot_marginal()`**: Plot marginal posterior for a target node
4. **`plot_stream()`**: Plot parent contributions to a child node
5. **`plot_idag()`**: Create animated heatmap of dynamic parameters

### Group analysis (`mdmp.group_analysis`)

Group-level tools live under **`mdmp.group_analysis`**: **VTS** (Virtual Typical Subject — representative time series), **IS** (Individual Structure — aggregate subject DAGs), and **distance** (pairwise subject dissimilarity / group-structure method). The root package also re-exports the main entry points.

**Virtual Typical Subject (VTS)** — multi-subject multivariate time series:

```python
from mdmp import compute_vts, MDM

# Data: list of (T_s x N) arrays, 3D (I x k x N), or DataFrame with subject_id
result = compute_vts(data, method="mean")      # Mean-based: avg per subject, then across
result = compute_vts(data, method="median")    # Median-based: robust to outliers across subjects
result = compute_vts(data, method="concatenation")  # Concatenate along time

# Use VTS with MDM
model = MDM(result.vts_data, method="hc")
```

**Individual Structure (IS) aggregation** — one global DAG from subject adjacency matrices.
Implementation lives under `mdmp.group_analysis.inds`.

```python
from mdmp import aggregate_individual_structures, plot_arcs, plot_dag

# Adjacency-only → consensus DAG (no Monte Carlo, no pooled Filt)
result = aggregate_individual_structures(list_of_adj_mats, tau=0.5, mc_n_samples=0)
fig = plot_dag(result)

# Fitted MDMs → consensus DAG + Monte Carlo on G* + pooled Filt for plot_arcs
result = aggregate_individual_structures(list_of_mdm_models, tau=0.5)
# Optional: mc_n_jobs=-1 parallelizes MC over time steps; mc_refit_n_jobs for refit/smoothing
fig2 = plot_arcs(result, plot_type="connections")
# result.global_beta_mc — inferential edge coefficients; not the same as pooled Filt
```

**Group-structure (GS) distance** — per-subject MDM estimates and pairwise dissimilarity for clustering / embedding (`mdmp.group_analysis.distance`):

```python
from mdmp import fit_individual_structures, compute_mdm_distance, plot_group_embedding

inds = fit_individual_structures(subjects)          # stage 1
dist = compute_mdm_distance(inds)                   # stages 2–3
fig = plot_group_embedding(dist, technique="nmds")    # stage 5
```

See `examples/04_vts_usage.py`, `notebooks/05-is-aggregation.ipynb`, `notebooks/08-gs-distance-projection.ipynb`, `notebooks/04-is-vs-vts-multi-individual.ipynb`, and `notebooks/09-gs-clusters-then-vts-is.ipynb` (GS clustering then VTS/IS per cluster on retail data).

## API Reference

### Core Functions

#### `mdmp.MDM`

Main model class for fitting MDM models.

#### `mdmp.dlm.dlm_filter()`

Core DLM filtering function.

#### `mdmp.dlm.dlm_smooth()`

Core DLM smoothing function.

#### `mdmp.scoring.select_discount_factors()`

Select optimal discount factors for each node.

#### `mdmp.structure.StructureLearner`

Structure learning algorithms.

## Documentation

Full documentation is available in the docstrings. To view:

```python
import mdmp
help(mdmp.MDM)
```

Release notes live in [`CHANGELOG.md`](CHANGELOG.md). Contributor setup, tests,
linting, and release steps are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

GPL-3.0

## Citation

If you use this package in your research, please cite:

```
@software{mdmp,
  title = {MDMP: Bayesian Dynamic Regression Model for Python},
  author = {Oliveira dos Santos, Matheus Augusto},
  version = {0.6.2},
  url = {https://github.com/maods2/mdmp}
}
```

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
development install, tests, linting, and how we cut releases.

## Acknowledgments

MDMP implements the Bayesian dynamic regression (MDM) approach for multivariate time series.

- **MDM model**: [Lilia Costa](mailto:liliacosta@ufba.br) and collaborators developed the MDM methodology
- **MDMP author**: [Matheus Augusto Oliveira dos Santos](mailto:matheusaugusto@ufba.br) — Python implementation and maintenance

