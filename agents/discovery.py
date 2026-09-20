"""
Autonomous Discovery Agent.

The headline feature: instead of waiting for "why did revenue fall?",
this scans the dataset itself and surfaces the 2-3 most significant
patterns automatically - period-over-period changes, and segments
(region/product/category) that deviate significantly from the rest.

Every finding here is a real, reproducible calculation on the actual
data - nothing here is AI-generated, which matters for the Verification
Agent downstream: it can always re-run these exact same calculations
and get the same numbers back.
"""

import pandas as pd


def discover(df: pd.DataFrame, key_columns: dict) -> list:
    findings = []

    metric_col = key_columns.get("primary_metric")
    time_col = key_columns.get("time_column")
    geo_col = key_columns.get("geo_column")
    category_cols = key_columns.get("category_columns", [])

    if not metric_col or metric_col not in df.columns:
        return findings

    clean_df = df[df[metric_col].notna()].copy()

    if time_col and time_col in clean_df.columns:
        clean_df[time_col] = pd.to_datetime(clean_df[time_col], errors="coerce")
        period_finding = _check_period_change(clean_df, metric_col, time_col)
        if period_finding:
            findings.append(period_finding)

    if geo_col and geo_col in clean_df.columns:
        segment_finding = _check_segment_deviation(clean_df, metric_col, geo_col, "region")
        if segment_finding:
            findings.append(segment_finding)

    for cat_col in category_cols[:2]:
        if cat_col in clean_df.columns:
            segment_finding = _check_segment_deviation(clean_df, metric_col, cat_col, "category")
            if segment_finding:
                findings.append(segment_finding)

    return findings[:3]


def _check_period_change(df: pd.DataFrame, metric_col: str, time_col: str) -> dict:
    valid = df.dropna(subset=[time_col])
    if len(valid) < 20:
        return None

    valid = valid.sort_values(time_col)
    midpoint = valid[time_col].min() + (valid[time_col].max() - valid[time_col].min()) / 2

    first_half = valid[valid[time_col] < midpoint][metric_col].sum()
    second_half = valid[valid[time_col] >= midpoint][metric_col].sum()

    if first_half == 0:
        return None

    pct_change = round(100 * (second_half - first_half) / first_half, 1)

    if abs(pct_change) < 8:
        return None

    direction = "dropped" if pct_change < 0 else "increased"
    return {
        "id": "period_change",
        "title": f"{metric_col} {direction} {abs(pct_change)}% between the first and second half of the period",
        "metric": metric_col,
        "magnitude": abs(pct_change),
        "direction": "down" if pct_change < 0 else "up",
        "evidence": {
            "first_half_total": round(float(first_half), 2),
            "second_half_total": round(float(second_half), 2),
            "pct_change": pct_change,
        },
    }


def _check_segment_deviation(df: pd.DataFrame, metric_col: str, segment_col: str, label: str) -> dict:
    grouped = df.groupby(segment_col)[metric_col].agg(["sum", "mean", "count"])
    if len(grouped) < 2:
        return None

    overall_mean = df[metric_col].mean()
    grouped["deviation_pct"] = round(100 * (grouped["mean"] - overall_mean) / overall_mean, 1)

    worst = grouped["deviation_pct"].idxmin()
    worst_row = grouped.loc[worst]

    if abs(worst_row["deviation_pct"]) < 15:
        return None

    return {
        "id": f"segment_deviation_{segment_col}",
        "title": f"{worst} has {abs(worst_row['deviation_pct'])}% {'lower' if worst_row['deviation_pct'] < 0 else 'higher'} average {metric_col} than other {label} segments",
        "metric": metric_col,
        "segment_column": segment_col,
        "segment_value": str(worst),
        "magnitude": abs(worst_row["deviation_pct"]),
        "direction": "down" if worst_row["deviation_pct"] < 0 else "up",
        "evidence": {
            "segment_average": round(float(worst_row["mean"]), 2),
            "overall_average": round(float(overall_mean), 2),
            "segment_record_count": int(worst_row["count"]),
        },
    }
