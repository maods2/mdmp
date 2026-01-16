# MDMP: Bayesian Network Modeling for Dynamic Multivariate Time Series

**MDMP** is a Python package for learning Bayesian network structures from multivariate time series and estimating time-varying dynamic parameters using Kalman filtering and smoothing. This package is a Python port of the R package **mdmr**.

## Features

- **Structure Learning**: Learn Bayesian network structures from multivariate time series using various algorithms (hill-climbing, tabu search, etc.)
- **Dynamic Parameter Estimation**: Estimate time-varying parameters using Kalman filtering and smoothing
- **Discount Factor Selection**: Automatically select optimal discount factors for each node
- **Visualization**: Comprehensive plotting tools for DAG structures, dynamic parameters, marginal posteriors, and animated heatmaps

## Installation

```bash
pip install mdmp
```

Or install from source with uv:

```bash
git clone https://github.com/arzevedo/mdmp.git
cd mdmp
uv venv
uv sync
source .venv/bin/activate
python -m pip install -e .
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- networkx >= 2.6.0

## Quick Start

### Basic Usage

```python
import numpy as np
import pandas as pd
from mdmp import MDM, plot_dag, plot_arcs

# Load or create your time series data
# Data should be T (time points) x N (variables)
data = pd.read_csv("your_timeseries.csv")

# Fit MDM model
model = MDM(data, method="hc", verbose=True)

# Plot DAG structure
plot_dag(model, plot_type="graph")

# Plot dynamic parameters
plot_arcs(model, plot_type="connections", distribution="filt")
```

### Example with Sample Data

```python
import numpy as np
from mdmp import MDM, plot_dag, plot_arcs, plot_marginal

# Generate sample data
np.random.seed(42)
T = 200  # Time points
N = 4    # Variables
data = np.random.randn(T, N)
data = pd.DataFrame(data, columns=[f"V{i+1}" for i in range(N)])

# Fit model
model = MDM(data, method="hc", nbf=15)

# Visualize results
plot_dag(model, plot_type="graph")
plot_dag(model, plot_type="heatmap")
plot_arcs(model, plot_type="connections", distribution="filt")
plot_marginal(model, target_node=0, distribution="filt")
```

## Main Components

### MDM Class

The main `MDM` class coordinates structure learning, discount factor selection, filtering, and smoothing:

```python
model = MDM(
    data,              # Time series data (pd.DataFrame or np.ndarray)
    method="hc",        # Structure learning method: "hc", "tabu", etc.
    nbf=15,            # Burn-in time point
    delta=None,        # Discount factor sequence (auto if None)
    verbose=True       # Print progress
)
```

**Attributes:**
- `adj_mat`: Adjacency matrix of learned DAG structure
- `data`: Original input data
- `DF`: Discount factor estimation results
- `Filt`: Filtered dynamic parameters
- `Smoo`: Smoothed dynamic parameters
- `node_names`: Names of variables/nodes

### Structure Learning Methods

Available methods for structure learning:
- `"hc"`: Hill-climbing (default)
- `"tabu"`: Tabu search

### Plotting Functions

1. **`plot_dag()`**: Plot DAG structure as graph or heatmap
2. **`plot_arcs()`**: Plot dynamic parameters over time
3. **`plot_marginal()`**: Plot marginal posterior for a target node
4. **`plot_stream()`**: Plot parent contributions to a child node
5. **`plot_idag()`**: Create animated heatmap of dynamic parameters

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
4. **Structure Learning**: Currently implements hill-climbing and tabu search natively

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
  author = {MDMP Contributors},
  version = {0.6.2},
  url = {https://github.com/arzevedo/mdmp}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

This package is a Python port of the R package **mdmr** by Lilia Costa and Arthur R. Azevedo.

## Changelog

### 0.6.2 (Initial Release)
- Ported core functionality from R package mdmr
- Implemented DLM filtering and smoothing
- Implemented MDM structure learning (hill-climbing, tabu search)
- Implemented plotting functions
- Added comprehensive documentation

