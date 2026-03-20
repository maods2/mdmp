# MDMP Examples

This directory contains example scripts demonstrating how to use the MDMP package for Bayesian network structure learning and dynamic parameter estimation.

## Example Files

### 01_basic_usage.py
**Basic MDM Usage**

A simple example showing:
- Creating synthetic time series data
- Fitting an MDM model with default parameters
- Accessing and inspecting results:
  - Adjacency matrix (learned DAG structure)
  - Discount factors
  - Filtered and smoothed parameters

**Run it:**
```bash
python examples/01_basic_usage.py
```

### 02_structure_learning.py
**Structure Learning Methods**

Demonstrates different structure learning methods:
- Hill-climbing (default, fast)
- Tabu search (often finds better structures)

Compares the structures learned by each method and shows how to use different learning algorithms.

**Run it:**
```bash
python examples/02_structure_learning.py
```

### 03_plotting.py
**Visualization Functions**

Comprehensive demonstration of all plotting functions:
- `plot_dag()` - Visualize DAG structure (graph and heatmap views)
- `plot_arcs()` - Plot dynamic parameters over time
- `plot_marginal()` - Plot marginal posterior for a target node
- `plot_stream()` - Show parent contributions to a child node
- `plot_idag()` - Create animated heatmap of dynamic parameters

All plots are saved to the `plot_examples/` directory.

**Run it:**
```bash
python examples/03_plotting.py
```

**Note:** Requires matplotlib. Some plots may require additional dependencies for animation.

### 04_advanced_usage.py
**Advanced Features**

Demonstrates advanced usage scenarios:
- Custom discount factor sequences
- Working with DataFrame input and custom node names
- Accessing filtered vs smoothed parameters
- Error handling
- Custom structure learning parameters
- Programmatic access to results

**Run it:**
```bash
python examples/04_advanced_usage.py
```

### 04_vts_usage.py
**Virtual Typical Subject (VTS)**

Demonstrates multi-subject aggregation for computing a representative subject:
- Creating multi-subject data (list, 3D array, DataFrame)
- Mean-based VTS: average per subject, then across subjects
- Concatenation-based VTS: stack along time for pooled MDM
- Comparing methods and evaluating representation quality
- Using VTS output with MDM for structure learning

**Run it:**
```bash
python examples/04_vts_usage.py
```

## Prerequisites

All examples use synthetic data and don't require external data files. However, they do require:

- Python 3.8+
- numpy
- pandas
- matplotlib (for plotting examples)
- mdmp package installed

To install mdmp in development mode:
```bash
pip install -e .
```

Or install with optional dependencies for hill-climbing:
```bash
pip install -e ".[hc]"
```

## Running All Examples

To run all examples in sequence:

```bash
cd examples
python 01_basic_usage.py
python 02_structure_learning.py
python 03_plotting.py
python 04_vts_usage.py
```

## Notes

- All examples use a fixed random seed (42) for reproducibility
- Examples use relatively small datasets for fast execution
- For plotting examples, figures are saved to files (non-interactive backend)
- Adjust parameters (data size, iterations, etc.) as needed for your use case

## Further Reading

- See the main [README.md](../README.md) for package overview
- Check the API documentation in function docstrings
- See `examples/example_usage.py` in the parent directory for a more complete example with real data processing
