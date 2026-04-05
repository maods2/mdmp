# MDMP: Bayesian Network Modeling for Dynamic Multivariate Time Series

**MDMP** is a Python package for learning Bayesian network structures from multivariate time series and estimating time-varying dynamic parameters using Kalman filtering and smoothing. It integrates structure learning algorithms including hill-climbing and tabu search with Kalman filtering and smoothing to estimate time-varying parameters for each node.

This package is a Python port of the R package **mdmr** developed by [Lilia Costa](mailto:liliacosta@ufba.br) and maintained by [Arthur R. Azevedo](mailto:arthur.rios@ufba.br).

## Features

- **Structure Learning**: Learn Bayesian network structures from multivariate time series using various algorithms (hill-climbing, tabu search, Max-Min Hill-Climbing)
- **Dynamic Parameter Estimation**: Estimate time-varying parameters using Kalman filtering and smoothing
- **Discount Factor Selection**: Automatically select optimal discount factors for each node
- **Parallel Processing**: Support for multiprocessing to speed up computation on multi-core systems
- **Progress Tracking**: Visual progress bars for long-running operations (when `verbose=True`)
- **Performance Logging**: Automatic timing and logging of total processing time
- **Visualization**: Comprehensive plotting tools for DAG structures, dynamic parameters, marginal posteriors, and animated heatmaps
- **Virtual Typical Subject (VTS)**: Compute a representative subject from multi-subject time series via concatenation-based or mean-based aggregation; compare methods and integrate with MDM

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

### Install from GitHub

You can install directly from the GitHub repository:

```bash
pip install git+https://github.com/maods2/mdmp.git
```

Or install a specific branch or tag:

```bash
pip install git+https://github.com/maods2/mdmp.git@main
pip install git+https://github.com/maods2/mdmp.git@v0.6.2
```

### Install from Source (Development)

Clone the repository and install in development mode:

```bash
# Clone the repository
git clone https://github.com/maods2/mdmp.git
cd mdmp

# Create virtual environment (choose one method)
# Method 1: Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Method 2: Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# Method 3: Using existing virtual environment (e.g., .venv-win)
.venv-win\Scripts\activate  # Windows
# or
source .venv/bin/activate   # Linux/Mac

# Install package in editable/development mode
pip install -e .

# Optional: Install with development dependencies
pip install -e ".[dev]"

# Optional: Install with hill-climbing support (requires pgmpy)
pip install -e ".[hc]"
```

### Install from Source (Production)

For production installation (non-editable):

```bash
git clone https://github.com/maods2/mdmp.git
cd mdmp

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in production mode
pip install .
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- networkx >= 2.6.0

### Optional Dependencies

- **pgmpy** (>=0.1.25): Required for hill-climbing structure learning method
- **notears**: Required for NOTEARS structure learning method (not on PyPI; install from GitHub, see below)
- **pytest** (>=7.0.0): For running unit tests
- **pytest-cov**: For test coverage reports (development only)

#### Installing NOTEARS (from GitHub)

The NOTEARS library is not available on PyPI. Install it from GitHub to use the `method="notears"` structure learning option:

```bash
# Install from official repository
pip install git+https://github.com/xunzheng/notears.git

# Or install from a local clone (e.g., if notears is in the same repo)
pip install -e ../notears
```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) `uv` package manager for faster dependency management

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/maods2/mdmp.git
   cd mdmp
   ```

2. **Create and activate a virtual environment:**
   
   **Using `uv` (recommended):**
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   uv sync
   ```
   
   **Using `venv`:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   pip install -e ".[dev]"
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

   Or with all development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Running Unit Tests

The package uses `pytest` for testing. To run all tests:

```bash
# Activate virtual environment first
source .venv/bin/activate  # Linux/Mac
# or
.venv-win\Scripts\activate  # Windows (if using .venv-win)

# Run all tests
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_scoring.py -v
pytest tests/test_structure.py -v

# Run tests with coverage (if pytest-cov is installed)
pytest tests/ -v --cov=mdmp --cov-report=term-missing

# Run tests without coverage (if pytest-cov is not installed)
pytest tests/ -v -o addopts=
```

**Note:** If you encounter errors about coverage options and `pytest-cov` is not installed, use the `-o addopts=` flag to override the configuration, or install pytest-cov:

```bash
pip install pytest-cov
```

### Test Structure

The test suite includes:
- **test_dlm.py**: Tests for DLM filtering and smoothing
- **test_mdm.py**: Tests for MDM class and main functionality
- **test_scoring.py**: Tests for scoring functions and discount factor selection
- **test_structure.py**: Tests for structure learning algorithms
- **test_utils.py**: Tests for utility functions
- **test_parallel.py**: Tests for parallel processing functionality
- **test_progress.py**: Tests for progress bar functionality
- **test_plotting.py**: Tests for plotting functions

### Code Quality and Linting

The project uses **Ruff** for fast Python linting and code quality checks. Ruff is configured in `pyproject.toml` and checks for:

- **E, W**: pycodestyle errors and warnings
- **F**: Pyflakes (unused imports, undefined names, etc.)
- **I**: Import sorting (isort)
- **B**: flake8-bugbear (common bugs and design problems)
- **C4**: flake8-comprehensions (better list/dict comprehensions)
- **UP**: pyupgrade (modernize Python syntax)

#### Basic Ruff Commands

```bash
# Activate virtual environment first
source .venv/bin/activate  # Linux/Mac
# or
.venv-win\Scripts\activate  # Windows

# Check for linting issues (read-only)
ruff check .

# Check and automatically fix issues
ruff check --fix .

# Check specific directories
ruff check mdmp/
ruff check tests/

# Check only import-related issues
ruff check . --select I

# Check only style issues
ruff check . --select E,W

# Show statistics
ruff check . --statistics
```

#### Ruff Configuration

The Ruff configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py38"
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501", "B008"]
```

**Key settings:**
- `line-length = 100`: Matches Black formatter line length
- `target-version = "py38"`: Ensures Python 3.8+ compatibility
- `E501` is ignored: Line length is handled by Black
- `B008` is ignored: Function calls in argument defaults are sometimes necessary

#### Formatting

The project uses **Black** for code formatting. Ruff can also format code, but Black is the primary formatter:

```bash
# Format code with Black
black mdmp/ tests/

# Check formatting without changing files
black --check mdmp/ tests/
```

#### Pre-commit Hooks (Optional)

If you have pre-commit configured, Ruff can run automatically:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

#### Common Ruff Workflow

1. **Before committing:**
   ```bash
   # Check for issues
   ruff check .
   
   # Auto-fix what can be fixed
   ruff check --fix .
   
   # Format code
   black mdmp/ tests/
   ```

2. **For specific issues:**
   ```bash
   # Fix only import sorting
   ruff check --fix . --select I
   
   # Check specific file
   ruff check mdmp/mdm.py
   ```

3. **In CI/CD:**
   ```bash
   # Fail if there are issues
   ruff check . --output-format=github
   ```
- **test_plotting.py**: Tests for plotting functions

All tests should pass before submitting pull requests.

## Example Usage

This walkthrough demonstrates how to use `MDMP` to learn a dynamic Bayesian network and visualize the results. This example uses a multivariate time series data [200 x 4].

### 1. Load the package and sample data

```python
import numpy as np
import pandas as pd
from mdmp import MDM, plot_dag, plot_arcs, plot_marginal, plot_stream, plot_idag, list_datasets, load_dataset

# Load sample data
data = load_dataset("covid_regional_timeseries")
print(data.head())
```

### 2. Fit the MDM model

```python
model = MDM(data, method="hc", verbose=True, n_jobs=-1)
# Learning structure using method: hc
# Selecting discount factors...
# Computing filtered estimates...
# Computing smoothed estimates...
# 
# MDM processing completed in 6.47 seconds
```

- `model` is an object of class `MDM` containing:
  - The inferred DAG structure (`adj_mat`)
  - Filtering and smoothing estimates (`Filt`, `Smoo`)
  - Local scores and optimization metadata
  - Available methods: `"hc"` (hill-climbing), `"tabu"` (tabu search), `"mmhc"` (Max-Min Hill-Climbing)

---

## Visualizations

### 3.1 DAG Structure

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

### 3.2 Arcs Over Time

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

### 3.3 Dynamic Heatmap Animation

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

### 3.4 Stream Plot for a Node

Plot contribution of parents to the dynamic evolution of a target node:

```python
plot_stream(mdm_object=model, child_node=0, distribution="filt")
```

![Stream Plot](plot_examples/stream_plot_Y1.png)

> Useful for assessing how different parents dynamically influence a given node.

- The `mdm_object` argument requires an object of class `MDM` as returned by `MDM()`.
- The `child_node` argument is an integer index of the target node (i.e., the child whose parents' effects are shown).
- The `distribution` argument defines whether to use filtered (`"filt"`) or smoothed (`"smoo"`) posterior estimates. Default is `"filt"`.

### 3.5 Marginal Posterior for a Node

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

- **`"hc"`**: **Hill-climbing** (default, fast) - Uses pgmpy's `HillClimbSearch` with custom MDM scoring function. Optimizes the log predictive likelihood per node. Requires `pgmpy` (install with: `pip install mdmp[hc]`).

- **`"mmhc"`**: **Max-Min Hill-Climbing** - First learns an undirected skeleton via MMPC (Max-Min Parents and Children), then orients edges using hill-climbing with custom MDM score. Requires `pgmpy` (install with: `pip install mdmp[hc]`).

- **`"tabu"`**: **Tabu search** - Uses pgmpy's HillClimbSearch with tabu search enabled (via `tabu_length` parameter). Requires `pgmpy` (install with: `pip install mdmp[hc]`). Supports `tabu_length` parameter (default: 100), `max_iter` (default: 1000000), `epsilon` (default: 0.0001), and other pgmpy HillClimbSearch parameters via `**kwargs`.

**Example with tabu search:**
```python
model = MDM(data, method="tabu", tabu_length=50, max_iter=1000, verbose=True)
```

**Registered but not yet implemented:**
- **`"ipa"`**: Integer Programming Approach using GOBNILP - Registered but currently raises `NotImplementedError`. The R package supports this via GOBNILP integration. Future Python implementation would require GOBNILP binary installation.

**Not yet implemented:**
- **`"h2pc"`**: H2PC algorithm - Available in R package via `bnlearn::h2pc`
- **`"rsmax2"`**: RSMAX2 algorithm - Available in R package via `bnlearn::rsmax2`

**Note:** These methods correspond to the algorithms available in the original R package `mdmr`, which uses `bnlearn::hc`, `bnlearn::tabu`, `bnlearn::mmhc`, `bnlearn::h2pc`, `bnlearn::rsmax2`, and supports GOBNILP for IPA. The Python implementation uses `pgmpy` as the backend for structure learning algorithms.

### Plotting Functions

1. **`plot_dag()`**: Plot DAG structure as graph or heatmap
2. **`plot_arcs()`**: Plot dynamic parameters over time
3. **`plot_marginal()`**: Plot marginal posterior for a target node
4. **`plot_stream()`**: Plot parent contributions to a child node
5. **`plot_idag()`**: Create animated heatmap of dynamic parameters

### Virtual Typical Subject (VTS)

For multi-subject multivariate time series, compute a representative subject:

```python
from mdmp import compute_vts, MDM

# Data: list of (T_s x N) arrays, 3D (I x k x N), or DataFrame with subject_id
result = compute_vts(data, method="mean")      # Mean-based: avg per subject, then across
result = compute_vts(data, method="concatenation")  # Concatenate along time

# Use VTS with MDM
model = MDM(result.vts_data, method="hc")
```

See `examples/04_vts_usage.py` and `notebooks/04_vts.ipynb` for full examples.

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

## Comparison with R Package

This Python package (`mdmp`) is a port of the R package `mdmr`. The main differences:

1. **Pythonic API**: Uses Python conventions and type hints
2. **Modular Design**: Code is organized into logical modules
3. **Modern Dependencies**: Uses NumPy, Pandas, Matplotlib instead of R equivalents
4. **Structure Learning**: Implements hill-climbing, tabu search, and Max-Min Hill-Climbing via pgmpy integration

## Documentation

Full documentation is available in the docstrings. To view:

```python
import mdmp
help(mdmp.MDM)
```

## License

GPL-3.0 (same as the original R package)

## Citation

If you use this package in your research, please cite:

```
@software{mdmp,
  title = {MDMP: Bayesian Dynamic Regression Model for Python},
  author = {},
  version = {0.6.2},
  url = {https://github.com/maods2/mdmp},
  note = {Python port of the mdmr R package}
}
```

For the original R package, please cite:

```
@software{mdmr,
  title = {mdmr: Bayesian dynamic regression model (MDM)},
  author = {Costa, Lilia and Azevedo, Arthur R.},
  version = {0.6.2},
  url = {https://github.com/arzevedo/mdmr}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

This package is a Python port of the R package **mdmr**.

- **Original Author**: [Lilia Costa](mailto:liliacosta@ufba.br) - Creator of the MDM model and original R package implementation
- **R Package Maintainer**: [Arthur R. Azevedo](mailto:arthur.rios@ufba.br) - Maintainer of the [mdmr R package](https://github.com/arzevedo/mdmr)
- **Python Port Maintainer**: [Matheus Augusto Oliveira dos Santos](mailto:matheusaugusto@ufba.br) - Responsible for adapting and implementing the Python version

## Changelog

### 0.7.0 (Current Version)
- Ported core functionality from R package mdmr
- Implemented DLM filtering and smoothing
- Implemented MDM structure learning (hill-climbing, tabu search, Max-Min Hill-Climbing)
- Implemented plotting functions
- Added comprehensive documentation
- **Parallel Processing**: Added multiprocessing support for discount factor selection, filtering, and smoothing
- **Progress Bars**: Integrated `tqdm` for visual progress tracking during long operations
- **Performance Logging**: Added automatic timing and logging of total processing time
- **Code Modularization**: Refactored code for better maintainability and testability
- **Tabu Search**: Fully implemented tabu search algorithm with configurable parameters

