"""
Simulate DAG structures for MDM evaluation.

This module provides functions to simulate synthetic time series data
from specified DAG structures for evaluating MDMP and MDMR structure learning.

Based on Capítulo 5 - Estudos de Simulação:
- Figura 5: 3-variable chain DAG (Y1 -> Y2 -> Y3)
- Figura 6: 5-variable DAG (Y1 root, Y1->Y2, Y1->Y3, Y2->Y4, Y3->Y4, Y2->Y5)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Default parameter values from Capítulo 5
W_VALUES = (0.0001, 0.01, 100.0)
V_VALUES = (0.01, 100.0)
T_VALUES = (100, 200)


def _simulate_dag_3var_single(
    seed: int, n: int, V: float, W: float
) -> Dict:
    """Single run of 3-variable DAG simulation (internal helper)."""
    np.random.seed(seed)

    n_n = 3
    m_ad = np.zeros((n_n, n_n), dtype=int)
    m_ad[0, 1] = 1  # Y1 -> Y2
    m_ad[1, 2] = 1  # Y2 -> Y3

    node_names = ['Y1', 'Y2', 'Y3']
    p = 5  # 1 intercept + 2 (Y2) + 2 (Y3)

    theta_ant = np.zeros(p)
    y = np.zeros((n, n_n))

    for z in range(n):
        theta_i = theta_ant + np.random.normal(0, np.sqrt(W), p)
        y_1i = theta_i[0] + np.random.normal(0, np.sqrt(V))
        y_2i = theta_i[1] + theta_i[2] * y_1i + np.random.normal(0, np.sqrt(V))
        y_3i = theta_i[3] + theta_i[4] * y_2i + np.random.normal(0, np.sqrt(V))
        y[z, :] = [y_1i, y_2i, y_3i]
        theta_ant = theta_i

    data = pd.DataFrame(y, columns=node_names)
    adj_df = pd.DataFrame(m_ad, index=node_names, columns=node_names)
    return {
        'data': data,
        'adjacency': m_ad,
        'adjacency_df': adj_df,
        'connection_info': None,
        'seed': seed,
    }


def simulate_dag_3var(
    seed: int = 1564,
    n: int = 200,
    V: float = 100.0,
    W: float = 0.1,
    n_individuals: int = 1,
    base_seed: Optional[int] = None,
) -> Union[Dict, List[Dict]]:
    """
    Simulate 3-variable DAG structure (Figura 5).

    DAG structure: Y1 -> Y2 -> Y3 (chain)
    Equations:
        θt = θt−1 + wt; wt ~ N(0,W)
        Yt(1) = θt(1)(1) + vt(1)
        Yt(2) = θt(1)(2) + θt(2)(2)Yt(1) + vt(2)
        Yt(3) = θt(1)(3) + θt(2)(3)Yt(2) + vt(3)

    Parameters
    ----------
    seed : int, optional
        Random seed (used when n_individuals=1). Default is 1564.
    n : int, optional
        Sample size (number of time points). Default is 200.
    V : float, optional
        Observational variance. Default is 100.0.
    W : float, optional
        System variance. Default is 0.1.
    n_individuals : int, optional
        Number of individuals to simulate (same topology, different seeds).
        Default is 1.
    base_seed : int, optional
        Base seed for individuals. Individual i uses base_seed + i.
        If None, uses seed when n_individuals=1, else seed + i.

    Returns
    -------
    dict or list of dict
        If n_individuals==1: single dict with data, adjacency, adjacency_df.
        If n_individuals>1: list of dicts, one per individual.
    """
    if base_seed is None:
        base_seed = seed

    if n_individuals == 1:
        return _simulate_dag_3var_single(seed=base_seed, n=n, V=V, W=W)

    results = []
    for i in range(n_individuals):
        ind_seed = base_seed + i
        res = _simulate_dag_3var_single(seed=ind_seed, n=n, V=V, W=W)
        results.append(res)
    return results


def _simulate_dag_5var_single(
    seed: int, n: int, V: float, W: float
) -> Dict:
    """Single run of 5-variable DAG simulation (internal helper)."""
    np.random.seed(seed)

    n_n = 5
    m_ad = np.zeros((n_n, n_n), dtype=int)
    m_ad[0, 1] = 1  # Y1 -> Y2
    m_ad[0, 2] = 1  # Y1 -> Y3
    m_ad[1, 3] = 1  # Y2 -> Y4
    m_ad[2, 3] = 1  # Y3 -> Y4
    m_ad[1, 4] = 1  # Y2 -> Y5

    node_names = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']
    p = 10

    theta_ant = np.zeros(p)
    y = np.zeros((n, n_n))

    for z in range(n):
        theta_i = theta_ant + np.random.normal(0, np.sqrt(W), p)
        y_1i = theta_i[0] + np.random.normal(0, np.sqrt(V))
        y_2i = theta_i[1] + theta_i[2] * y_1i + np.random.normal(0, np.sqrt(V))
        y_3i = theta_i[3] + theta_i[4] * y_1i + np.random.normal(0, np.sqrt(V))
        y_4i = theta_i[5] + theta_i[6] * y_2i + theta_i[7] * y_3i + np.random.normal(0, np.sqrt(V))
        y_5i = theta_i[8] + theta_i[9] * y_2i + np.random.normal(0, np.sqrt(V))
        y[z, :] = [y_1i, y_2i, y_3i, y_4i, y_5i]
        theta_ant = theta_i

    data = pd.DataFrame(y, columns=node_names)
    adj_df = pd.DataFrame(m_ad, index=node_names, columns=node_names)
    return {
        'data': data,
        'adjacency': m_ad,
        'adjacency_df': adj_df,
        'connection_info': None,
        'seed': seed,
    }


def simulate_dag_5var(
    seed: int = 1564,
    n: int = 200,
    V: float = 100.0,
    W: float = 0.1,
    n_individuals: int = 1,
    base_seed: Optional[int] = None,
) -> Union[Dict, List[Dict]]:
    """
    Simulate 5-variable DAG structure (Figura 6).

    Equations:
        θt = θt−1 + wt; wt ~ N(0,W),
        Yt(1) = θt(1)(1) + vt(1),
        Yt(2) = θt(1)(2) + θt(2)(2)Yt(1) + vt(2),
        Yt(3) = θt(1)(3) + θt(2)(3)Yt(1) + vt(3),
        Yt(4) = θt(1)(4) + θt(2)(4)Yt(2) + θt(3)(4)Yt(3) + vt(4),
        Yt(5) = θt(1)(5) + θt(2)(5)Yt(2) + vt(5);
        vt(r) ~ N(0, V), r = 1, ..., 5.

    DAG structure:
        Y1 (intercept only)
        Y1 -> Y2, Y1 -> Y3
        Y2 -> Y4, Y3 -> Y4
        Y2 -> Y5

    Parameters
    ----------
    seed : int, optional
        Random seed (used when n_individuals=1). Default is 1564.
    n : int, optional
        Sample size (number of time points). Default is 200.
    V : float, optional
        Observational variance. Default is 100.0.
    W : float, optional
        System variance. Default is 0.1.
    n_individuals : int, optional
        Number of individuals to simulate (same topology, different seeds).
        Default is 1.
    base_seed : int, optional
        Base seed for individuals. Individual i uses base_seed + i.
        If None, uses seed when n_individuals=1, else seed + i.

    Returns
    -------
    dict or list of dict
        If n_individuals==1: single dict with data, adjacency, adjacency_df.
        If n_individuals>1: list of dicts, one per individual.
    """
    if base_seed is None:
        base_seed = seed

    if n_individuals == 1:
        return _simulate_dag_5var_single(seed=base_seed, n=n, V=V, W=W)

    results = []
    for i in range(n_individuals):
        ind_seed = base_seed + i
        res = _simulate_dag_5var_single(seed=ind_seed, n=n, V=V, W=W)
        results.append(res)
    return results


def run_simulations(
    output_dir: str = "data/",
    base_seed: int = 1564,
    n_individuals: int = 1,
    w_values: Optional[Tuple[float, ...]] = None,
    v_values: Optional[Tuple[float, ...]] = None,
    t_values: Optional[Tuple[int, ...]] = None,
    write_legacy_names: bool = False,
) -> None:
    """
    Run simulations for both DAGs across all W, V, T combinations.

    Generates files for each (W, V, T) combination and each individual.
    File naming: dag_{3var|5var}_W{w}_V{v}_T{t}_ind{i}.csv

    Parameters
    ----------
    output_dir : str, optional
        Output directory for CSV files. Default is "data/".
    base_seed : int, optional
        Base seed for individuals. Individual i uses base_seed + i.
        Default is 1564.
    n_individuals : int, optional
        Number of individuals per (W, V, T) combination. Default is 1.
    w_values : tuple of float, optional
        System variance values. Default is (0.0001, 0.01, 100.0).
    v_values : tuple of float, optional
        Observational variance values. Default is (0.01, 100.0).
    t_values : tuple of int, optional
        Sample size values. Default is (100, 200).
    write_legacy_names : bool, optional
        If True and exactly one (W,V,T) combination with n_individuals=1,
        also write dag_3var_simulated.csv, dag_5var_simulated.csv and
        their adjacency files for notebook compatibility. Default is False.
    """
    import os

    w_vals = w_values if w_values is not None else W_VALUES
    v_vals = v_values if v_values is not None else V_VALUES
    t_vals = t_values if t_values is not None else T_VALUES

    os.makedirs(output_dir, exist_ok=True)
    single_combo = len(w_vals) == 1 and len(v_vals) == 1 and len(t_vals) == 1
    write_legacy = write_legacy_names and single_combo and n_individuals == 1

    for W in w_vals:
        for V in v_vals:
            for T in t_vals:
                prefix_3 = f"dag_3var_W{W}_V{V}_T{T}"
                prefix_5 = f"dag_5var_W{W}_V{V}_T{T}"

                # 3-variable DAG (Figura 5)
                results_3 = simulate_dag_3var(
                    seed=base_seed,
                    n=T,
                    V=V,
                    W=W,
                    n_individuals=n_individuals,
                    base_seed=base_seed,
                )
                if n_individuals == 1:
                    results_3 = [results_3]
                for i, res in enumerate(results_3):
                    fname = f"{prefix_3}_ind{i + 1}.csv"
                    res['data'].to_csv(os.path.join(output_dir, fname), index=False)
                results_3[0]['adjacency_df'].to_csv(
                    os.path.join(output_dir, f"{prefix_3}_true_adjacency.csv")
                )
                if write_legacy:
                    results_3[0]['data'].to_csv(
                        os.path.join(output_dir, "dag_3var_simulated.csv"), index=False)
                    results_3[0]['adjacency_df'].to_csv(
                        os.path.join(output_dir, "dag_3var_true_adjacency.csv"))

                # 5-variable DAG (Figura 6)
                results_5 = simulate_dag_5var(
                    seed=base_seed,
                    n=T,
                    V=V,
                    W=W,
                    n_individuals=n_individuals,
                    base_seed=base_seed,
                )
                if n_individuals == 1:
                    results_5 = [results_5]
                for i, res in enumerate[Dict | List[Dict]](results_5):
                    fname = f"{prefix_5}_ind{i + 1}.csv"
                    res['data'].to_csv(os.path.join(output_dir, fname), index=False)
                results_5[0]['adjacency_df'].to_csv(
                    os.path.join(output_dir, f"{prefix_5}_true_adjacency.csv")
                )
                if write_legacy:
                    results_5[0]['data'].to_csv(
                        os.path.join(output_dir, "dag_5var_simulated.csv"), index=False)
                    results_5[0]['adjacency_df'].to_csv(
                        os.path.join(output_dir, "dag_5var_true_adjacency.csv"))

    print(f"Simulations complete. Output: {output_dir}")
    print(f"  Combinations: W={w_vals}, V={v_vals}, T={t_vals}")
    print(f"  Individuals per combination: {n_individuals}")


if __name__ == "__main__":
    # Run simulations: single default scenario with legacy filenames for notebooks
    run_simulations(
        output_dir="simulation/multi-individual/",
        n_individuals=5,
        w_values=(0.01,),
        v_values=(100.0,),
        t_values=(200,),
        write_legacy_names=True,
    )
