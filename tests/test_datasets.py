"""Tests for bundled dataset loaders and retail helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mdmp.datasets import list_datasets, load_dataset
from mdmp.retail import (
    _BUNDLED_CSV,
    DAG_LABELS,
    aggregate_by_level,
    food_group_subjects,
    load_retail,
    parse_retail_dataset,
)


def test_list_datasets_includes_retail():
    names = list_datasets()
    assert "mdmr_test_data" in names
    assert "covid_regional_timeseries" in names
    assert "retail" in names


def test_load_dataset_known_shapes():
    mdmr = load_dataset("mdmr_test_data")
    assert isinstance(mdmr, pd.DataFrame)
    assert mdmr.shape[1] == 4

    covid = load_dataset("covid_regional_timeseries")
    assert isinstance(covid, pd.DataFrame)
    assert covid.shape[1] == 5


def test_load_dataset_retail_sales_only():
    sales = load_dataset("retail")
    assert isinstance(sales, pd.DataFrame)
    assert sales.shape == (180, 16)
    assert sales.columns[0] == "Time"
    assert len([c for c in sales.columns if c != "Time"]) == 15


def test_load_dataset_unknown_name():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("not_a_dataset")


def test_load_retail_hierarchy_and_sales():
    sales, hierarchy = load_retail()
    assert sales.shape == (180, 16)
    assert list(hierarchy.columns) == ["type", "line", "item"]
    assert len(hierarchy) == 15
    assert hierarchy.index.tolist() == [c for c in sales.columns if c != "Time"]


def test_parse_retail_dataset_is_load_retail():
    assert parse_retail_dataset is load_retail


def test_load_retail_custom_path():
    sales, hierarchy = load_retail(Path(_BUNDLED_CSV))
    assert sales.shape == (180, 16)
    assert len(hierarchy) == 15


def test_aggregate_by_level_type_has_english_columns():
    sales, hierarchy = load_retail()
    c3 = aggregate_by_level(sales, hierarchy, "type")
    assert c3.shape[1] == 7
    english = {
        "Alcoholic beverages",
        "Basic cereals",
        "Bakery",
        "Preserves",
        "Dairy",
        "Non-alcoholic beverages",
        "Biscuits and confectionery",
    }
    assert set(c3.columns) == english
    assert english.issubset(set(DAG_LABELS))


def test_food_group_subjects_default_partition():
    sales, _ = load_retail()
    subjects, subject_ids, node_names, membership = food_group_subjects(sales)
    assert len(subjects) == 5
    assert len(subject_ids) == 5
    assert len(node_names) == 3
    assert all(s.shape == (180, 3) for s in subjects)
    assert set(membership) == set(subject_ids)
