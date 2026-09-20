"""
What-If Simulation Agent.

Estimates the effect of a hypothetical change (e.g. "increase discount
by 10%") using a real linear regression fitted on the actual historical
data - not a made-up number. Confidence is reported based on the
regression's R-squared, so a weak historical relationship honestly
shows LOW confidence instead of a falsely precise answer.

Supports two scenario types out of the box:
  - discount_change: estimates revenue/volume impact of a discount shift
  - customer_loss: estimates revenue impact of losing top N customers
"""

import pandas as pd
import numpy as np


def simulate_discount_change(df: pd.DataFrame, key_columns: dict, discount_change_pct: float) -> dict:
    metric_col = key_columns.get("primary_metric")
    if not metric_col or "Discount" not in df.columns:
        return {"error": "Dataset doesn't have both a revenue metric and a Discount column - cannot simulate this."}

    clean = df[[metric_col, "Discount"]].dropna()
    if len(clean) < 30:
        return {"error": "Not enough data points to build a reliable simulation."}

    x = clean["Discount"].values
    y = clean[metric_col].values

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    current_avg_discount = x.mean()
    new_discount = current_avg_discount + (discount_change_pct / 100)

    current_predicted_revenue = slope * current_avg_discount + intercept
    new_predicted_revenue = slope * new_discount + intercept

    revenue_change_pct = round(100 * (new_predicted_revenue - current_predicted_revenue) / current_predicted_revenue, 1) if current_predicted_revenue != 0 else 0
    revenue_change_pct = float(revenue_change_pct)

    confidence = _confidence_label(r_squared)

    return {
        "scenario": f"Increase average discount by {discount_change_pct} percentage points",
        "current_avg_discount": round(float(current_avg_discount), 3),
        "simulated_avg_discount": round(float(new_discount), 3),
        "estimated_revenue_change_pct": revenue_change_pct,
        "r_squared": round(float(r_squared), 3),
        "confidence": confidence,
        "disclaimer": "This is an estimate based on the historical relationship between discount and revenue in this dataset, not a guarantee. It assumes the same relationship holds outside the range of data seen so far.",
    }


def simulate_customer_loss(df: pd.DataFrame, key_columns: dict, top_n: int) -> dict:
    metric_col = key_columns.get("primary_metric")
    id_col = None
    for col in df.columns:
        if "id" in col.lower():
            id_col = col
            break

    if not metric_col or not id_col:
        return {"error": "Dataset doesn't have both a revenue metric and a customer identifier - cannot simulate this."}

    clean = df[[id_col, metric_col]].dropna()
    by_customer = clean.groupby(id_col)[metric_col].sum().sort_values(ascending=False)

    if len(by_customer) < top_n:
        return {"error": f"Dataset only has {len(by_customer)} unique customers, fewer than the {top_n} requested."}

    total_revenue = by_customer.sum()
    top_customers_revenue = by_customer.head(top_n).sum()
    impact_pct = round(100 * top_customers_revenue / total_revenue, 1) if total_revenue != 0 else 0
    impact_pct = float(impact_pct)

    return {
        "scenario": f"Lose top {top_n} customers by revenue",
        "total_revenue": round(float(total_revenue), 2),
        "at_risk_revenue": round(float(top_customers_revenue), 2),
        "impact_pct_of_total": impact_pct,
        "confidence": "HIGH",
        "disclaimer": "This is a direct calculation from historical revenue, not a prediction - it shows exposure, assuming these customers contribute at the same rate going forward.",
    }


def _confidence_label(r_squared: float) -> str:
    if r_squared >= 0.5:
        return "HIGH"
    if r_squared >= 0.2:
        return "MEDIUM"
    return "LOW"
