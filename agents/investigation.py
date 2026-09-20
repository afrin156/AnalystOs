"""
WHY / Investigation Agent.

Given a finding from the Discovery Agent, this drills into the data to
find what else changed alongside it - other columns that moved in a
correlated way. It never claims causation; it ranks contributing
factors by how strongly they co-occur with the finding, and explicitly
labels them as "possible factors," matching the evidence-first
philosophy of the whole project.

Bedrock is used only to phrase the final explanation in plain language -
every number quoted in that explanation comes from the contributing
factors this agent calculates first.
"""

import pandas as pd
from bedrock_client import ask_bedrock


def investigate(df: pd.DataFrame, finding: dict, key_columns: dict) -> dict:
    metric_col = finding["metric"]
    contributing_factors = []

    category_cols = key_columns.get("category_columns", [])
    geo_col = key_columns.get("geo_column")
    status_cols = key_columns.get("status_columns", [])

    candidate_cols = [c for c in (category_cols + [geo_col] + status_cols) if c and c in df.columns]

    if finding["id"].startswith("segment_deviation"):
        segment_col = finding["segment_column"]
        segment_val = finding["segment_value"]
        subset = df[df[segment_col] == segment_val]
        rest = df[df[segment_col] != segment_val]

        for col in candidate_cols:
            if col == segment_col:
                continue
            factor = _compare_column(subset, rest, col, metric_col)
            if factor:
                contributing_factors.append(factor)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            if col == metric_col:
                continue
            factor = _compare_numeric_column(df, finding, col, metric_col)
            if factor:
                contributing_factors.append(factor)

    elif finding["id"] == "period_change":
        time_col = key_columns.get("time_column")
        if time_col and time_col in df.columns:
            contributing_factors.extend(
                _find_segments_driving_period_change(df, finding, metric_col, time_col, candidate_cols)
            )

    contributing_factors.sort(key=lambda f: abs(f.get("strength", 0)), reverse=True)
    top_factors = contributing_factors[:4]

    explanation = _explain_with_ai(finding, top_factors)

    return {
        "finding_id": finding["id"],
        "contributing_factors": top_factors,
        "explanation": explanation,
        "disclaimer": "These are correlated factors found in the data, not confirmed causes. Establishing causation would require controlled analysis beyond what this dataset alone can show.",
    }


def _find_segments_driving_period_change(df: pd.DataFrame, finding: dict, metric_col: str, time_col: str, candidate_cols: list) -> list:
    """
    For a period-change finding, checks each candidate segment column to
    see which specific segment's before/after change is disproportionately
    large compared to the overall change - i.e. which segment is driving
    the aggregate trend, rather than the change being spread evenly.
    """
    factors = []
    valid = df.dropna(subset=[time_col, metric_col]).copy()
    valid[time_col] = pd.to_datetime(valid[time_col], errors="coerce")
    valid = valid.dropna(subset=[time_col])
    if len(valid) < 20:
        return factors

    midpoint = valid[time_col].min() + (valid[time_col].max() - valid[time_col].min()) / 2
    overall_change_pct = finding["evidence"]["pct_change"]

    for col in candidate_cols:
        if col not in valid.columns:
            continue
        for segment_val in valid[col].dropna().unique():
            segment_df = valid[valid[col] == segment_val]
            if len(segment_df) < 15:
                continue
            first = segment_df[segment_df[time_col] < midpoint][metric_col].sum()
            second = segment_df[segment_df[time_col] >= midpoint][metric_col].sum()
            if first == 0:
                continue
            segment_pct_change = round(100 * (second - first) / first, 1)

            if abs(segment_pct_change - overall_change_pct) > 15 and abs(segment_pct_change) > abs(overall_change_pct):
                factors.append({
                    "column": col,
                    "type": "segment_period_change",
                    "detail": f"'{col} = {segment_val}' changed {segment_pct_change}% across the period, more than the overall {overall_change_pct}% - a likely driver of the overall trend",
                    "strength": abs(segment_pct_change),
                })

    return factors


def _compare_column(subset: pd.DataFrame, rest: pd.DataFrame, col: str, metric_col: str) -> dict:
    if subset[col].dtype == bool or subset[col].dropna().isin([0, 1, True, False]).all():
        subset_rate = subset[col].mean()
        rest_rate = rest[col].mean()
        if rest_rate == 0:
            return None
        diff_pct = round(100 * (subset_rate - rest_rate) / rest_rate, 1)
        if abs(diff_pct) < 15:
            return None
        return {
            "column": col,
            "type": "rate_comparison",
            "detail": f"'{col}' rate is {abs(diff_pct)}% {'higher' if diff_pct > 0 else 'lower'} in this segment than elsewhere",
            "strength": abs(diff_pct),
        }
    return None


def _compare_numeric_column(df: pd.DataFrame, finding: dict, col: str, metric_col: str) -> dict:
    if finding["id"].startswith("segment_deviation"):
        segment_col = finding["segment_column"]
        segment_val = finding["segment_value"]
        subset = df[df[segment_col] == segment_val]
        rest = df[df[segment_col] != segment_val]

        subset_mean = subset[col].mean()
        rest_mean = rest[col].mean()
        if rest_mean == 0 or pd.isna(subset_mean) or pd.isna(rest_mean):
            return None
        diff_pct = round(100 * (subset_mean - rest_mean) / rest_mean, 1)
        if abs(diff_pct) < 15:
            return None
        return {
            "column": col,
            "type": "average_comparison",
            "detail": f"Average '{col}' is {abs(diff_pct)}% {'higher' if diff_pct > 0 else 'lower'} in this segment ({round(subset_mean,3)} vs {round(rest_mean,3)} elsewhere)",
            "strength": abs(diff_pct),
        }
    return None


def _explain_with_ai(finding: dict, factors: list) -> str:
    if not factors:
        return "No strongly correlated factors were found in the available columns for this finding."

    factors_text = "\n".join([f"- {f['detail']}" for f in factors])

    system_prompt = """You are a careful data analyst explaining a finding to a business audience.
Rules: use ONLY the facts given below. Never claim certainty about causation - use words like
"may have contributed", "is associated with", "possible factor". Keep it to 2-3 sentences, plain language, no jargon."""

    user_message = f"""Finding: {finding['title']}

Correlated factors found in the data:
{factors_text}

Write a short, evidence-based explanation a business manager could understand."""

    return ask_bedrock(system_prompt, user_message)
