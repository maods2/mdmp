# MDMP/MDMR Simulation and Comparison

This directory contains simulation scripts and notebooks for evaluating and comparing MDMP (Python) and MDMR (R) structure learning algorithms on synthetic DAG structures.

## Overview

This simulation framework generates synthetic time series data from known DAG structures, applies both MDMP and MDMR hill-climbing algorithms, and compares their performance using various evaluation metrics.

## Directory Structure

```
mdmp/simulation/
├── README.md                          # This file
├── simulate_dags.py                   # Python script to generate synthetic DAG data
├── simulation_utils.R                 # R utility functions for metrics computation
├── 04_mdmp_structure_learning.ipynb   # Python notebook: MDMP structure learning
├── 05_mdmr_structure_learning.ipynb   # R notebook: MDMR structure learning
├── 06_comparison_metrics.ipynb        # Python notebook: Compare MDMP vs MDMR
└── data/                              # Generated simulation data (created automatically)
    ├── dag_4var_simulated.csv
    ├── dag_4var_true_adjacency.csv
    ├── dag_5var_simulated.csv
    ├── dag_5var_true_adjacency.csv
    └── ... (result files)
```

## Workflow

Follow these steps in order:

### 1. Generate Simulated Data

Run the Python simulation script to generate synthetic time series data:

```bash
python simulate_dags.py
```

This will create:
- `data/dag_4var_simulated.csv` - 4-variable time series data
- `data/dag_4var_true_adjacency.csv` - True adjacency matrix for 4-variable DAG
- `data/dag_5var_simulated.csv` - 5-variable time series data
- `data/dag_5var_true_adjacency.csv` - True adjacency matrix for 5-variable DAG

### 2. Run MDMP Structure Learning

Execute the Python notebook `04_mdmp_structure_learning.ipynb` to:
- Load simulated data
- Apply MDMP hill-climbing structure learning
- Compute evaluation metrics
- Save results to `data/mdmp_results.csv`

### 3. Run MDMR Structure Learning

Execute the R notebook `05_mdmr_structure_learning.ipynb` to:
- Load simulated data
- Apply MDMR hill-climbing using the `mdmr` package
- Compute evaluation metrics
- Save results to `data/mdmr_results.csv`

**Note:** This notebook requires the `mdmr` R package to be installed and uses R kernel in Jupyter.

### 4. Compare Results

Execute the Python notebook `06_comparison_metrics.ipynb` to:
- Load results from both MDMP and MDMR
- Create comparison tables
- Generate visualizations (bar charts, heatmaps, time comparisons)

## Simulated DAG Structures

### 4-Variable DAG

```
Y3 -> Y1 -> Y2
Y3 -> Y4 -> Y2
```

- **Y3**: Root node (no parents)
- **Y1, Y4**: Children of Y3
- **Y2**: Child of both Y1 and Y4

### 5-Variable DAG

```
Y1 -> Y2 -> Y4
Y1 -> Y3 -> Y4
Y2 -> Y5
```

- **Y1**: Root node (no parents)
- **Y2, Y3**: Children of Y1
- **Y4**: Child of both Y2 and Y3
- **Y5**: Child of Y2

## Files Description

### `simulate_dags.py`

Python script containing functions to simulate DAG structures:
- `simulate_dag_4var()` - Generates 4-variable DAG time series
- `simulate_dag_5var()` - Generates 5-variable DAG time series
- `run_simulations()` - Main function to run simulations and save CSV files

**Parameters:**
- `seed`: Random seed for reproducibility (default: 1564)
- `n`: Sample size / number of time points (default: 200)
- `V`: Observational variance (default: 100.0)
- `W`: System variance (default: 0.1)

### `simulation_utils.R`

R utility functions for evaluation:
- `build_connection_matrix()` - Creates symmetric connection matrix from directed adjacency matrix
- `compute_metrics()` - Computes evaluation metrics (accuracy, sensitivity, specificity, PPV, NPV, directional accuracy)

### `04_mdmp_structure_learning.ipynb`

Python notebook for MDMP analysis:
- Uses `mdmp.MDM` class with hill-climbing method
- Computes metrics comparing estimated vs true DAG structure
- Saves results for comparison

### `05_mdmr_structure_learning.ipynb`

R notebook for MDMR analysis:
- Uses `mdmr::mdm()` function with `method = "hc"`
- Computes same metrics as MDMP notebook
- Saves results for comparison

### `06_comparison_metrics.ipynb`

Python notebook for comparison:
- Loads results from both MDMP and MDMR
- Creates side-by-side comparison tables
- Generates visualizations:
  - Bar charts comparing metrics
  - Heatmaps of adjacency matrices (true, MDMP, MDMR)
  - Computation time comparison

## Evaluation Metrics

Both notebooks compute the following metrics:

1. **Accuracy** - Overall connection accuracy
2. **Sensitivity** - True positive rate (detecting true connections)
3. **Specificity** - True negative rate (correctly identifying non-connections)
4. **PPV** (Positive Predictive Value) - Precision of detected connections
5. **NPV** (Negative Predictive Value) - Precision of non-connections
6. **Directional Accuracy** - Accuracy of edge directions

## Dependencies

### Python
- `mdmp` - MDM Python package
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

### R
- `mdmr` - MDM R package
- `bnlearn`
- `reshape2`
- `dplyr`

## Notes

- The simulation uses the same random seed (1564) by default for reproducibility
- All data files are saved in CSV format for compatibility between Python and R
- Results are saved in the `data/` directory for easy comparison
- The R notebook requires Jupyter with R kernel (IRkernel) installed

## Example Usage

```python
# Generate data
from simulate_dags import run_simulations
run_simulations(output_dir="./data/", seed_4var=1564, seed_5var=1564)

# Then run the notebooks in order: 04 -> 05 -> 06
```
