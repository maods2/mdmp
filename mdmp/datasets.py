"""
Dataset loading module for MDMP package.

This module provides convenient functions to load sample datasets
included with the package for testing and demonstration purposes.
"""

from pathlib import Path

import pandas as pd

from .retail import (
    C3_LABELS,
    C4_LABELS,
    DAG_LABELS,
    FOOD_GROUP_MEMBERSHIP,
    SKU_DAG_LABELS,
    aggregate_by_level,
    cohort_summary,
    english_group_label,
    food_group_subjects,
    load_retail,
    monthly_subjects,
    one_sku_per_type,
    order_skus_by_level,
    parse_retail_dataset,
    product_lag_subjects,
)

__all__ = [
    "list_datasets",
    "load_dataset",
    "load_mdmr_test_data",
    "load_covid_regional_timeseries",
    "load_retail",
    "parse_retail_dataset",
    "aggregate_by_level",
    "cohort_summary",
    "english_group_label",
    "food_group_subjects",
    "monthly_subjects",
    "one_sku_per_type",
    "order_skus_by_level",
    "product_lag_subjects",
    "C3_LABELS",
    "C4_LABELS",
    "DAG_LABELS",
    "FOOD_GROUP_MEMBERSHIP",
    "SKU_DAG_LABELS",
]


def _get_data_dir() -> Path:
    """
    Get the path to the data directory.

    Tries multiple locations:
    1. Inside the mdmp package directory (for installed packages)
    2. Relative to package location (for development at root)
    3. Relative to current working directory (for development)

    Returns
    -------
    Path
        Path to the data directory.

    Raises
    ------
    FileNotFoundError
        If the data directory cannot be found.
    """
    # Try inside mdmp package directory (for installed packages)
    mdmp_data_dir = Path(__file__).parent / "data"
    if mdmp_data_dir.exists():
        return mdmp_data_dir

    # Try relative to package location (for development at root)
    package_dir = Path(__file__).parent.parent
    data_dir = package_dir / "data"
    if data_dir.exists():
        return data_dir

    # Try relative to current working directory (for development)
    cwd_data_dir = Path.cwd() / "data"
    if cwd_data_dir.exists():
        return cwd_data_dir

    raise FileNotFoundError(
        "Could not find data directory. Please ensure the 'data' folder exists "
        "inside the mdmp package directory, at the package root, or in the current working directory."
    )


def list_datasets() -> list[str]:
    """
    List all available datasets.

    Returns
    -------
    list of str
        List of dataset names that can be loaded.
    """
    return [
        "mdmr_test_data",
        "covid_regional_timeseries",
        "retail",
    ]


def load_mdmr_test_data() -> pd.DataFrame:
    """
    Load the MDMR test dataset.

    This dataset contains test time series data with variables: Y1, Y2, Y3, Y4.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the MDMR test data with shape (T, 4).

    Examples
    --------
    >>> from mdmp.datasets import load_mdmr_test_data
    >>> data = load_mdmr_test_data()
    >>> print(data.shape)
    >>> print(data.head())
    """
    data_dir = _get_data_dir()
    file_path = data_dir / "mdmr_test_data.csv"
    return pd.read_csv(file_path)


def load_covid_regional_timeseries() -> pd.DataFrame:
    """
    Load the COVID regional timeseries dataset.

    This dataset contains regional COVID-19 data with variables:
    Southeast, Midwest, Northeast, North, South

    Returns
    -------
    pd.DataFrame
        DataFrame containing the COVID regional timeseries data.

    Examples
    --------
    >>> from mdmp.datasets import load_covid_regional_timeseries
    >>> data = load_covid_regional_timeseries()
    >>> print(data.shape)
    >>> print(data.head())
    """
    data_dir = _get_data_dir()
    file_path = data_dir / "covid_regional_timeseries.csv"
    return pd.read_csv(file_path)


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset to load. Available datasets can be listed
        using `list_datasets()`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested dataset.

        For ``"retail"`` this is the sales panel only (Time + SKU columns).
        Use :func:`load_retail` when you also need the product hierarchy.

    Raises
    ------
    ValueError
        If the dataset name is not recognized.

    Examples
    --------
    >>> from mdmp.datasets import load_dataset
    >>> data = load_dataset("mdmr_test_data")
    >>> print(data.head())
    """
    name = name.lower().strip()

    dataset_loaders = {
        "mdmr_test_data": load_mdmr_test_data,
        "covid_regional_timeseries": load_covid_regional_timeseries,
        "retail": lambda: load_retail()[0],
    }

    if name not in dataset_loaders:
        available = ", ".join(dataset_loaders.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        )

    return dataset_loaders[name]()
