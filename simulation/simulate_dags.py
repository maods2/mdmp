"""
Simulate DAG structures for MDM evaluation.

This module provides functions to simulate synthetic time series data
from specified DAG structures for evaluating MDMP and MDMR structure learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def simulate_dag_4var(seed: int = 1564, n: int = 200, V: float = 100.0, W: float = 0.1) -> Dict:
    """
    Simulate 4-variable DAG structure.
    
    Based on the structure from tcc-michel-mdm/Exemplo - MDM-Hill Climbing.R
    
    DAG structure:
        Y3 -> Y1 -> Y2
        Y3 -> Y4 -> Y2
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default is 1564.
    n : int, optional
        Sample size (number of time points). Default is 200.
    V : float, optional
        Observational variance. Default is 100.0.
    W : float, optional
        System variance. Default is 0.1.
    
    Returns
    -------
    dict
        Dictionary containing:
        - data: DataFrame with columns Y1, Y2, Y3, Y4
        - adjacency: True adjacency matrix (4 x 4) as numpy array
        - connection_info: Connection matrix information (for compatibility)
    """
    np.random.seed(seed)
    
    # Number of variables
    n_n = 4
    
    # Create true adjacency matrix
    # Rows = parents, Cols = children
    m_ad = np.zeros((n_n, n_n), dtype=int)
    m_ad[2, 0] = 1  # Y3 -> Y1 (0-indexed: Y3=2, Y1=0)
    m_ad[2, 3] = 1  # Y3 -> Y4 (0-indexed: Y3=2, Y4=3)
    m_ad[0, 1] = 1  # Y1 -> Y2 (0-indexed: Y1=0, Y2=1)
    m_ad[3, 1] = 1  # Y4 -> Y2 (0-indexed: Y4=3, Y2=1)
    
    # Add column and row names (will be used when converting to DataFrame)
    node_names = ['Y1', 'Y2', 'Y3', 'Y4']
    
    # Number of theta parameters (intercepts + edge coefficients)
    # 4 intercepts + 4 edges = 8 parameters
    p = 8
    
    # Initial theta (all zeros)
    theta_ant = np.zeros(p)
    
    # Initialize data matrix
    y = np.zeros((n, n_n))
    
    # Simulate time series
    for z in range(n):
        # System equation: theta_t = theta_{t-1} + w_t, w_t ~ N(0, W)
        theta_i = theta_ant + np.random.normal(0, np.sqrt(W), p)
        
        # Observation equations following DAG structure:
        # Y3 (no parents, only intercept) - index 2
        y_3i = theta_i[0] + np.random.normal(0, np.sqrt(V))
        
        # Y1 (parent: Y3) - index 0
        y_1i = theta_i[1] + theta_i[2] * y_3i + np.random.normal(0, np.sqrt(V))
        
        # Y4 (parent: Y3) - index 3
        y_4i = theta_i[3] + theta_i[4] * y_3i + np.random.normal(0, np.sqrt(V))
        
        # Y2 (parents: Y1, Y4) - index 1
        y_2i = theta_i[5] + theta_i[6] * y_1i + theta_i[7] * y_4i + np.random.normal(0, np.sqrt(V))
        
        # Store data (order: Y1, Y2, Y3, Y4)
        y[z, :] = [y_1i, y_2i, y_3i, y_4i]
        
        # Update theta for next iteration
        theta_ant = theta_i
    
    # Convert to DataFrame
    data = pd.DataFrame(y, columns=node_names)
    
    # Create adjacency matrix with names for compatibility
    adj_df = pd.DataFrame(m_ad, index=node_names, columns=node_names)
    
    return {
        'data': data,
        'adjacency': m_ad,
        'adjacency_df': adj_df,
        'connection_info': None  # Can be computed if needed
    }


def simulate_dag_5var(seed: int = 1564, n: int = 200, V: float = 100.0, W: float = 0.1) -> Dict:
    """
    Simulate 5-variable DAG structure.
    
    Based on the provided equations:
    θt = θt−1 + wt; wt ~ N(0,W),
    Yt(1) = θt(1)(1) + vt(1),
    Yt(2) = θt(1)(2) + θt(2)(2)Yt(1) + vt(2),
    Yt(3) = θt(1)(3) + θt(2)(3)Yt(1) + vt(3),
    Yt(4) = θt(1)(4) + θt(2)(4)Yt(2) + θt(3)(4)Yt(3) + vt(4),
    Yt(5) = θt(1)(5) + θt(2)(5)Yt(2) + vt(5);
    vt(r) ~ N(0, V), r = 1, ..., 5.
    
    DAG structure:
        Y1 (intercept only)
        Y1 -> Y2
        Y1 -> Y3
        Y2 -> Y4
        Y3 -> Y4
        Y2 -> Y5
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default is 1564.
    n : int, optional
        Sample size (number of time points). Default is 200.
    V : float, optional
        Observational variance. Default is 100.0.
    W : float, optional
        System variance. Default is 0.1.
    
    Returns
    -------
    dict
        Dictionary containing:
        - data: DataFrame with columns Y1, Y2, Y3, Y4, Y5
        - adjacency: True adjacency matrix (5 x 5) as numpy array
        - connection_info: Connection matrix information (for compatibility)
    """
    np.random.seed(seed)
    
    # Number of variables
    n_n = 5
    
    # Create true adjacency matrix
    # Rows = parents, Cols = children
    m_ad = np.zeros((n_n, n_n), dtype=int)
    m_ad[0, 1] = 1  # Y1 -> Y2 (0-indexed: Y1=0, Y2=1)
    m_ad[0, 2] = 1  # Y1 -> Y3 (0-indexed: Y1=0, Y3=2)
    m_ad[1, 3] = 1  # Y2 -> Y4 (0-indexed: Y2=1, Y4=3)
    m_ad[2, 3] = 1  # Y3 -> Y4 (0-indexed: Y3=2, Y4=3)
    m_ad[1, 4] = 1  # Y2 -> Y5 (0-indexed: Y2=1, Y5=4)
    
    # Add column and row names
    node_names = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']
    
    # Number of theta parameters:
    # Y1: 1 (intercept)
    # Y2: 2 (intercept + Y1)
    # Y3: 2 (intercept + Y1)
    # Y4: 3 (intercept + Y2 + Y3)
    # Y5: 2 (intercept + Y2)
    # Total: 1 + 2 + 2 + 3 + 2 = 10 parameters
    p = 10
    
    # Initial theta (all zeros)
    theta_ant = np.zeros(p)
    
    # Initialize data matrix
    y = np.zeros((n, n_n))
    
    # Simulate time series
    for z in range(n):
        # System equation: theta_t = theta_{t-1} + w_t, w_t ~ N(0, W)
        theta_i = theta_ant + np.random.normal(0, np.sqrt(W), p)
        
        # Observation equations following DAG structure:
        # Y1: intercept only - index 0
        y_1i = theta_i[0] + np.random.normal(0, np.sqrt(V))
        
        # Y2: intercept + Y1 - index 1
        y_2i = theta_i[1] + theta_i[2] * y_1i + np.random.normal(0, np.sqrt(V))
        
        # Y3: intercept + Y1 - index 2
        y_3i = theta_i[3] + theta_i[4] * y_1i + np.random.normal(0, np.sqrt(V))
        
        # Y4: intercept + Y2 + Y3 - index 3
        y_4i = theta_i[5] + theta_i[6] * y_2i + theta_i[7] * y_3i + np.random.normal(0, np.sqrt(V))
        
        # Y5: intercept + Y2 - index 4
        y_5i = theta_i[8] + theta_i[9] * y_2i + np.random.normal(0, np.sqrt(V))
        
        # Store data (order: Y1, Y2, Y3, Y4, Y5)
        y[z, :] = [y_1i, y_2i, y_3i, y_4i, y_5i]
        
        # Update theta for next iteration
        theta_ant = theta_i
    
    # Convert to DataFrame
    data = pd.DataFrame(y, columns=node_names)
    
    # Create adjacency matrix with names for compatibility
    adj_df = pd.DataFrame(m_ad, index=node_names, columns=node_names)
    
    return {
        'data': data,
        'adjacency': m_ad,
        'adjacency_df': adj_df,
        'connection_info': None  # Can be computed if needed
    }


def run_simulations(output_dir: str = "../mdmp/simulation/data/", 
                   seed_4var: int = 1564, seed_5var: int = 1564,
                   n: int = 200, V: float = 100.0, W: float = 0.1):
    """
    Run simulations for both DAGs and save CSV files.
    
    Parameters
    ----------
    output_dir : str, optional
        Output directory for CSV files. Default is "../mdmp/data/simulated/".
    seed_4var : int, optional
        Random seed for 4-variable DAG. Default is 1564.
    seed_5var : int, optional
        Random seed for 5-variable DAG. Default is 1564.
    n : int, optional
        Sample size. Default is 200.
    V : float, optional
        Observational variance. Default is 100.0.
    W : float, optional
        System variance. Default is 0.1.
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Simulating 4-variable DAG...")
    # Simulate 4-variable DAG
    result_4var = simulate_dag_4var(seed=seed_4var, n=n, V=V, W=W)
    
    # Save data
    result_4var['data'].to_csv(
        os.path.join(output_dir, "dag_4var_simulated.csv"),
        index=False
    )
    
    # Save true adjacency matrix
    result_4var['adjacency_df'].to_csv(
        os.path.join(output_dir, "dag_4var_true_adjacency.csv")
    )
    
    print("4-variable DAG simulation complete.")
    print(f"  Data saved to: {os.path.join(output_dir, 'dag_4var_simulated.csv')}")
    print(f"  Adjacency saved to: {os.path.join(output_dir, 'dag_4var_true_adjacency.csv')}")
    
    print("\nSimulating 5-variable DAG...")
    # Simulate 5-variable DAG
    result_5var = simulate_dag_5var(seed=seed_5var, n=n, V=V, W=W)
    
    # Save data
    result_5var['data'].to_csv(
        os.path.join(output_dir, "dag_5var_simulated.csv"),
        index=False
    )
    
    # Save true adjacency matrix
    result_5var['adjacency_df'].to_csv(
        os.path.join(output_dir, "dag_5var_true_adjacency.csv")
    )
    
    print("5-variable DAG simulation complete.")
    print(f"  Data saved to: {os.path.join(output_dir, 'dag_5var_simulated.csv')}")
    print(f"  Adjacency saved to: {os.path.join(output_dir, 'dag_5var_true_adjacency.csv')}")
    
    print("\nAll simulations complete!")


if __name__ == "__main__":
    # Run simulations when script is executed directly
    run_simulations()
