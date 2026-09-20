"""
Data Understanding Agent.

Runs first, on every uploaded dataset. Its only job is to figure out
what each column IS before anything else touches the data - an
identifier, a financial metric, a time dimension, a geography, a
categorical business variable, or a status flag.

This is deliberately rule-based, not AI-based: column typing should be
fast, deterministic, and explainable, not something an LLM guesses at.
"""

import pandas as pd


def understand(df: pd.DataFrame) -> dict:
    columns_info = []

    for col in df.columns:
        series = df[col]
        role = _infer_role(col, series)
        columns_info.append({
            "name": col,
            "role": role,
            "dtype": str(series.dtype),
            "sample_values": series.dropna().head(3).astype(str).tolist(),
        })

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info,
    }


def _infer_role(col_name: str, series: pd.Series) -> str:
    name = col_name.lower()

    if "id" in name and series.nunique() > 0.9 * len(series):
        return "identifier"

    if any(k in name for k in ["date", "time", "created", "order_date"]):
        return "time_dimension"

    if any(k in name for k in ["revenue", "cost", "price", "amount", "discount", "profit", "sales"]):
        return "financial_metric"

    if any(k in name for k in ["region", "state", "city", "country", "location"]):
        return "geographic_dimension"

    if any(k in name for k in ["status", "return", "flag", "is_", "active"]):
        return "status_flag"

    if series.dtype == "object" and series.nunique() < max(20, 0.05 * len(series)):
        return "categorical_dimension"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric_metric"

    if pd.api.types.is_string_dtype(series) and series.nunique() < max(20, 0.05 * len(series)):
        return "categorical_dimension"

    return "unclassified"


def find_key_columns(df: pd.DataFrame, understanding: dict) -> dict:
    """
    Convenience helper other agents use: pulls out the single best
    column for each role, so they don't all re-implement this logic.
    """
    by_role = {}
    for col in understanding["columns"]:
        by_role.setdefault(col["role"], []).append(col["name"])

    return {
        "primary_metric": (by_role.get("financial_metric") or by_role.get("numeric_metric") or [None])[0],
        "time_column": (by_role.get("time_dimension") or [None])[0],
        "geo_column": (by_role.get("geographic_dimension") or [None])[0],
        "category_columns": by_role.get("categorical_dimension", []),
        "status_columns": by_role.get("status_flag", []),
    }
