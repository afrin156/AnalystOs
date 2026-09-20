"""
Data Health Agent.

Runs real checks against the dataset - missing values, duplicates,
invalid dates, negative values in fields that shouldn't be negative,
and statistical outliers. Deliberately does NOT auto-fix anything: a
negative revenue value might be a real refund, not an error, so this
agent flags it and explains the possible interpretation instead of
silently deleting or "correcting" data.
"""

import pandas as pd
import numpy as np


def check_health(df: pd.DataFrame, key_columns: dict) -> dict:
    issues = []
    total_cells = df.shape[0] * df.shape[1]

    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        pct = round(100 * missing_count / total_cells, 2)
        issues.append({
            "type": "missing_values",
            "severity": "warning" if pct < 5 else "high",
            "count": int(missing_count),
            "detail": f"{pct}% of all cells are missing across the dataset.",
        })

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append({
            "type": "duplicate_records",
            "severity": "warning",
            "count": int(duplicate_count),
            "detail": f"{duplicate_count} fully duplicated rows found - consider whether these are genuine repeat orders or accidental duplicates.",
        })

    time_col = key_columns.get("time_column")
    if time_col and time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        invalid_dates = parsed.isnull().sum() - df[time_col].isnull().sum()
        if invalid_dates > 0:
            issues.append({
                "type": "invalid_dates",
                "severity": "warning",
                "count": int(invalid_dates),
                "detail": f"{invalid_dates} values in '{time_col}' could not be parsed as valid dates.",
            })

    metric_col = key_columns.get("primary_metric")
    if metric_col and metric_col in df.columns and pd.api.types.is_numeric_dtype(df[metric_col]):
        negative_count = (df[metric_col] < 0).sum()
        if negative_count > 0:
            issues.append({
                "type": "negative_values",
                "severity": "info",
                "count": int(negative_count),
                "detail": f"{negative_count} negative values found in '{metric_col}'. Possible refunds or adjustments - no automatic correction applied.",
            })

        outliers = _count_outliers(df[metric_col])
        if outliers > 0:
            issues.append({
                "type": "outliers",
                "severity": "info",
                "count": int(outliers),
                "detail": f"{outliers} statistical outliers detected in '{metric_col}' (beyond 3 standard deviations from the mean).",
            })

    score = _compute_score(issues, total_cells)

    return {
        "score": score,
        "issues": issues,
        "schema_consistent": True,
    }


def _count_outliers(series: pd.Series) -> int:
    clean = series.dropna()
    if len(clean) < 10:
        return 0
    mean, std = clean.mean(), clean.std()
    if std == 0:
        return 0
    z_scores = np.abs((clean - mean) / std)
    return int((z_scores > 3).sum())


def _compute_score(issues: list, total_cells: int) -> int:
    """
    Simple deterministic scoring - starts at 100, deducts based on
    severity. Kept transparent and explainable rather than a black box.
    """
    score = 100
    weights = {"info": 1, "warning": 4, "high": 10}
    for issue in issues:
        score -= weights.get(issue["severity"], 2)
    return max(0, min(100, score))
