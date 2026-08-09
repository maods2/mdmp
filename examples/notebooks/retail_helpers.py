"""Retail CSV helpers for the MDMP library demo notebook.

Adapted from ``mdm-experiment/retail_loader.py`` and
``mdm-experiment/retail_aggregated_dags.py`` so the demo does not depend on
the experiment tree on ``sys.path``.
"""

# from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "MDM_retail_dataset.csv"

# Short English labels for individual SKU nodes (Graphviz / GS).
SKU_DAG_LABELS: Dict[str, str] = {
    "Cerveja|Lata": "Beer-can",
    "Cerveja|Latao": "Beer-large",
    "Cerveja|Retornavel 250ml": "Beer-ret",
    "Arroz|5kg": "Rice",
    "Feijao|1kg": "Beans",
    "Pao|Forma": "Bread",
    "Oleo_Soja|1L": "Soy oil",
    "Molho_Tomate|250g": "Tomato",
    "Leite|Integral": "Milk",
    "Iogurte|Natural": "Yogurt",
    "Refrigerante|Retornavel 1.5L": "Soda-1.5L",
    "Refrigerante|Lata": "Soda-can",
    "Suco_Natural|1L": "Juice",
    "Cereal_Barra|250g": "Cereal bar",
    "Chocolate|Leite 1kg": "Chocolate",
}

# C3 Product Line (hierarchy.type)
C3_LABELS: Dict[str, str] = {
    "Alcoolicas": "Alcoholic beverages",
    "Cereais Básicos": "Basic cereals",
    "Padaria": "Bakery",
    "Conservas": "Preserves",
    "Leite": "Dairy",
    "Não-alcoólicas": "Non-alcoholic beverages",
    "biscoitos, bomboniere": "Biscuits and confectionery",
}

# C4 Product Type (hierarchy.line)
C4_LABELS: Dict[str, str] = {
    "Cerveja": "Beer",
    "Arroz": "Rice",
    "Feijao": "Beans",
    "Pao": "Bread",
    "Oleo_Soja": "Soybean oil",
    "Molho_Tomate": "Tomato sauce",
    "Leite": "Milk",
    "Iogurte": "Yogurt",
    "Refrigerante": "Soft drinks",
    "Suco_Natural": "Natural juice",
    "Cereal_Barra": "Cereal bar",
    "Chocolate": "Chocolate",
}

GROUP_LABELS: Dict[str, str] = {**C3_LABELS, **C4_LABELS}

# Shorter labels for aggregated DAG nodes.
DAG_LABELS: Dict[str, str] = {
    "Alcoholic beverages": "Alcoholic",
    "Basic cereals": "Cereals",
    "Bakery": "Bakery",
    "Preserves": "Preserves",
    "Dairy": "Dairy",
    "Non-alcoholic beverages": "Non-alc.",
    "Biscuits and confectionery": "Biscuits",
    "Beer": "Beer",
    "Rice": "Rice",
    "Beans": "Beans",
    "Bread": "Bread",
    "Soybean oil": "Soy oil",
    "Tomato sauce": "Tomato",
    "Yogurt": "Yogurt",
    "Soft drinks": "Soft drinks",
    "Natural juice": "Juice",
    "Cereal bar": "Cereal bar",
    "Chocolate": "Chocolate",
    "Milk": "Milk",
}

# Fixed partition: equal N=3; Padaria joins Cereais / Leite / Conservas / biscoitos.
FOOD_GROUP_MEMBERSHIP: Dict[str, List[str]] = {
    "Alcoolicas": [
        "Cerveja|Lata",
        "Cerveja|Latao",
        "Cerveja|Retornavel 250ml",
    ],
    "Nao-alcoolicas": [
        "Refrigerante|Retornavel 1.5L",
        "Refrigerante|Lata",
        "Suco_Natural|1L",
    ],
    "Cereais_Padaria": [
        "Arroz|5kg",
        "Feijao|1kg",
        "Pao|Forma",
    ],
    "Leite_Conservas": [
        "Leite|Integral",
        "Iogurte|Natural",
        "Oleo_Soja|1L",
    ],
    "Conservas_Biscoitos": [
        "Molho_Tomate|250g",
        "Cereal_Barra|250g",
        "Chocolate|Leite 1kg",
    ],
}


def load_retail_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Read the Mac Roman CSV with no header (183 x 16)."""
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    return pd.read_csv(csv_path, header=None, encoding="mac_roman")


def parse_retail_dataset(
    path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse hierarchy + daily sales.

    Returns
    -------
    sales : DataFrame
        Columns Time + one column per SKU (Line|Item). Shape (180, 16).
    hierarchy : DataFrame
        Index = node_names; columns type, line, item.
    """
    raw = load_retail_raw(path)
    types = raw.iloc[0, 1:].astype(str).str.strip().tolist()
    lines = raw.iloc[1, 1:].astype(str).str.strip().tolist()
    items = raw.iloc[2, 1:].astype(str).str.strip().tolist()
    node_names = [f"{line}|{item}" for line, item in zip(lines, items)]

    hierarchy = pd.DataFrame(
        {"type": types, "line": lines, "item": items},
        index=node_names,
    )

    sales = raw.iloc[3:].copy()
    sales.columns = ["Time"] + node_names
    sales["Time"] = pd.to_datetime(sales["Time"], format="%m/%d/%y")
    for col in node_names:
        sales[col] = pd.to_numeric(sales[col], errors="raise")
    sales = sales.reset_index(drop=True)
    return sales, hierarchy


def english_group_label(value: str, labels: Optional[Dict[str, str]] = None) -> str:
    """Map Portuguese hierarchy keys to English labels (with ASCII fallback)."""
    mapping = labels if labels is not None else GROUP_LABELS
    if value in mapping:
        return mapping[value]
    normalized = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    normalized_labels = {
        unicodedata.normalize("NFKD", key)
        .encode("ascii", "ignore")
        .decode()
        .lower(): label
        for key, label in mapping.items()
    }
    if normalized.startswith("n") and "alco" in normalized:
        return "Non-alcoholic beverages"
    return normalized_labels.get(normalized, str(value))


def aggregate_by_level(
    sales: pd.DataFrame,
    hierarchy: pd.DataFrame,
    level_column: str,
) -> pd.DataFrame:
    """Sum SKU series within each hierarchy level; English column names."""
    labels = C3_LABELS if level_column == "type" else C4_LABELS
    node_cols = [c for c in sales.columns if c != "Time"]
    X = sales.set_index("Time")[node_cols]
    frames = []
    for group, skus in hierarchy.groupby(level_column, sort=False).groups.items():
        label = english_group_label(str(group), labels)
        frames.append(X[list(skus)].sum(axis=1).rename(label))
    return pd.concat(frames, axis=1)


def order_skus_by_level(
    hierarchy: pd.DataFrame,
    level_column: str,
) -> List[str]:
    """Return SKU names ordered by hierarchy group (C3 type or C4 line)."""
    if level_column not in {"type", "line"}:
        raise ValueError("level_column must be 'type' or 'line'")
    ordered: List[str] = []
    for _, skus in hierarchy.groupby(level_column, sort=False).groups.items():
        ordered.extend(list(skus))
    return ordered


def one_sku_per_type(hierarchy: pd.DataFrame) -> List[str]:
    """Pick the first SKU in each C3 type (shared nodes for monthly IS/VTS)."""
    return hierarchy.groupby("type", sort=False).head(1).index.tolist()


def monthly_subjects(
    sales: pd.DataFrame,
    node_names: Optional[Sequence[str]] = None,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Split the wide series into one (T_m x N) array per calendar month.

    Returns subjects, subject_ids (YYYY-MM), node_names.
    """
    if node_names is None:
        node_names = [c for c in sales.columns if c != "Time"]
    node_names = list(node_names)

    subjects: List[np.ndarray] = []
    subject_ids: List[str] = []
    months = sales["Time"].dt.to_period("M")
    for period, group in sales.groupby(months, sort=True):
        arr = group.loc[:, node_names].to_numpy(dtype=float)
        subjects.append(arr)
        subject_ids.append(str(period))
    return subjects, subject_ids, node_names


def food_group_subjects(
    sales: pd.DataFrame,
    membership: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[np.ndarray], List[str], List[str], Dict[str, List[str]]]:
    """
    Build one (T x N) subject per food group with a shared column count N.

    Default membership yields S=5, N=3, T=len(sales), covering all 15 SKUs.
    Generic node names SKU1..SKUN are used because product labels differ
    across subjects; ``membership`` keeps the real SKU mapping.
    """
    groups = membership if membership is not None else FOOD_GROUP_MEMBERSHIP
    n_per = {len(skus) for skus in groups.values()}
    if len(n_per) != 1:
        raise ValueError(f"all food groups must share the same N; got sizes {n_per}")
    n = next(iter(n_per))

    missing = [sku for skus in groups.values() for sku in skus if sku not in sales.columns]
    if missing:
        raise KeyError(f"SKUs not found in sales columns: {missing}")

    node_names = [f"SKU{i + 1}" for i in range(n)]
    subjects: List[np.ndarray] = []
    subject_ids: List[str] = []
    for sid, skus in groups.items():
        subjects.append(sales.loc[:, skus].to_numpy(dtype=float))
        subject_ids.append(sid)

    return subjects, subject_ids, node_names, dict(groups)


def product_lag_subjects(
    sales: pd.DataFrame,
    *,
    n_lags: int = 3,
    node_names: Optional[Sequence[str]] = None,
    subject_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    One subject per SKU (no cross-product aggregation).

    Each product series is turned into an (T - n_lags + 1) x n_lags panel of
    lag coordinates ``[y_t, y_{t-1}, ..., y_{t-n_lags+1}]`` so MDM/GS can run
    with a shared node count across products.
    """
    if n_lags < 2:
        raise ValueError(f"n_lags must be >= 2, got {n_lags}")

    skus = [c for c in sales.columns if c != "Time"]
    if subject_ids is None:
        subject_ids = list(skus)
    else:
        subject_ids = list(subject_ids)
        if len(subject_ids) != len(skus):
            raise ValueError("subject_ids length must match number of SKU columns")

    if node_names is None:
        node_names = [f"lag{i}" for i in range(n_lags)]
    else:
        node_names = list(node_names)
        if len(node_names) != n_lags:
            raise ValueError(f"node_names length must equal n_lags={n_lags}")

    subjects: List[np.ndarray] = []
    for sku in skus:
        y = sales[sku].to_numpy(dtype=float)
        panel = np.column_stack(
            [y[n_lags - 1 :]]
            + [y[n_lags - 1 - k : len(y) - k] for k in range(1, n_lags)]
        )
        subjects.append(panel)

    return subjects, list(subject_ids), node_names


def cohort_summary(
    subjects: Sequence[np.ndarray],
    subject_ids: Sequence[str],
    node_names: Sequence[str],
) -> Dict[str, object]:
    """Compact shape / T statistics for sanity checks."""
    lengths = [int(s.shape[0]) for s in subjects]
    return {
        "n_subjects": len(subjects),
        "n_nodes": len(node_names),
        "node_names": list(node_names),
        "subject_ids": list(subject_ids),
        "T_per_subject": lengths,
        "T_min": min(lengths) if lengths else 0,
        "T_max": max(lengths) if lengths else 0,
        "shapes": [tuple(s.shape) for s in subjects],
    }
