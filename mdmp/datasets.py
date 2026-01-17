"""
Dataset loading module for MDMP package.

This module provides convenient functions to load sample datasets
included with the package for testing and demonstration purposes.
"""

from pathlib import Path

import pandas as pd


def _get_data_dir() -> Path:
    """
    Get the path to the data directory.

    Tries multiple locations:
    1. Relative to package location (for installed packages)
    2. Relative to current working directory (for development)
    3. Relative to the mdmp package directory

    Returns
    -------
    Path
        Path to the data directory.

    Raises
    ------
    FileNotFoundError
        If the data directory cannot be found.
    """
    # Try relative to package location
    package_dir = Path(__file__).parent.parent
    data_dir = package_dir / "data"
    if data_dir.exists():
        return data_dir

    # Try relative to current working directory
    cwd_data_dir = Path.cwd() / "data"
    if cwd_data_dir.exists():
        return cwd_data_dir

    # Try relative to mdmp package directory
    mdmp_data_dir = Path(__file__).parent / "data"
    if mdmp_data_dir.exists():
        return mdmp_data_dir

    raise FileNotFoundError(
        "Could not find data directory. Please ensure the 'data' folder exists "
        "either at the package root or in the current working directory."
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
    }

    if name not in dataset_loaders:
        available = ", ".join(dataset_loaders.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        )

    return dataset_loaders[name]()
